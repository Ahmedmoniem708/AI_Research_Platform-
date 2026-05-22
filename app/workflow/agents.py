"""
AI Writing Workflow using LangGraph for the AI Knowledge Platform.
Implements a research workflow with Planner, Writer, and Reviewer agents.
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from app.rag.engine import RAGEngine
from app.generation.generation import LocalLLM
import logging

logger = logging.getLogger(__name__)

class WorkflowState(TypedDict):
    """State for the AI writing workflow."""
    user_prompt: str
    outline: List[str]
    drafted_sections: Dict[str, str]
    current_status: str

def PlannerAgent(state: WorkflowState, llm: LocalLLM) -> WorkflowState:
    """
    Planner Agent: Takes the user's prompt and uses the local LLM to generate
    a structured outline (a list of section headings).
    """
    logger.info("PlannerAgent: Generating outline for user prompt")

    prompt = f"""You are an expert academic planner. Given the user's research prompt,
    create a detailed outline with clear section headings for a comprehensive research response.

    User Prompt: {state['user_prompt']}

    Provide ONLY a list of section headings, one per line, without any additional text or numbering.
    Each heading should be a concise, descriptive title for a section of the research paper.

    Outline:"""

    try:
        # Generate outline using LLM
        sampling_params = llm.sampling_params
        sampling_params.max_tokens = 256
        sampling_params.temperature = 0.3

        outputs = llm.llm.generate([prompt], sampling_params)
        outline_text = outputs[0].outputs[0].text.strip()

        # Parse outline into list of sections
        outline = [line.strip() for line in outline_text.split('\n') if line.strip()]

        # Filter out any empty lines or non-heading lines
        outline = [section for section in outline if section and not section.startswith('.')]

        logger.info(f"PlannerAgent: Generated outline with {len(outline)} sections")

        return {
            **state,
            "outline": outline,
            "current_status": "outline_generated"
        }
    except Exception as e:
        logger.error(f"PlannerAgent failed: {str(e)}")
        return {
            **state,
            "outline": ["Introduction", "Main Content", "Conclusion"],  # Fallback outline
            "current_status": "outline_generated_fallback"
        }

def WriterAgent(state: WorkflowState, llm: LocalLLM, rag_engine: RAGEngine) -> WorkflowState:
    """
    Writer Agent: Iterates over the outline. For each section, it queries the
    rag_engine to get chunks (with citation tags), and uses the LLM to draft
    the text while forcing the inclusion of the exact citation tags.
    """
    logger.info("WriterAgent: Starting to draft sections")

    drafted_sections = state.get("drafted_sections", {})
    outline = state.get("outline", [])

    if not outline:
        logger.warning("WriterAgent: No outline provided, returning empty sections")
        return {
            **state,
            "drafted_sections": drafted_sections,
            "current_status": "writing_skipped"
        }

    try:
        for section_title in outline:
            logger.info(f"WriterAgent: Drafting section: {section_title}")

            # Query RAG engine for relevant chunks
            query = f"{state['user_prompt']} {section_title}"
            retrieved_chunks = rag_engine.query(query_text=query, top_k=5)

            if not retrieved_chunks:
                logger.warning(f"WriterAgent: No chunks retrieved for section: {section_title}")
                drafted_sections[section_title] = f"[Content for {section_title} would be generated based on retrieved sources]"
                continue

            # Prepare context with citation tags
            context_parts = []
            citation_instructions = []

            for i, chunk in enumerate(retrieved_chunks):
                metadata = chunk["metadata"]
                # Get the pre-built citation tag
                citation_tag = metadata.get("citation_tag", f"[Unknown Source]")
                context_parts.append(f"{citation_tag} {chunk['text']}")
                citation_instructions.append(f"Use citation tag: {citation_tag}")

            context = "\n\n".join(context_parts)
            citation_rules = "\n".join(citation_instructions)

            # Construct writing prompt
            writing_prompt = f"""You are an expert academic writer. Write a comprehensive section for a research paper based on the provided sources.

Section Title: {section_title}
Overall Research Topic: {state['user_prompt']}

You MUST write using ONLY the information provided in the context below.
For every factual statement, you MUST include the exact citation tag from the source material.
Do not add information that is not present in the sources.

{citation_rules}

Context:
{context}

Write the section content for "{section_title}":
"""

            # Generate section content
            sampling_params = llm.sampling_params
            sampling_params.max_tokens = 1024
            sampling_params.temperature = 0.5

            outputs = llm.llm.generate([writing_prompt], sampling_params)
            section_content = outputs[0].outputs[0].text.strip()

            # Store the drafted section
            drafted_sections[section_title] = section_content
            logger.info(f"WriterAgent: Completed section: {section_title}")

        return {
            **state,
            "drafted_sections": drafted_sections,
            "current_status": "sections_drafted"
        }

    except Exception as e:
        logger.error(f"WriterAgent failed: {str(e)}")
        return {
            **state,
            "drafted_sections": drafted_sections,
            "current_status": "writing_failed"
        }

def ReviewerAgent(state: WorkflowState) -> WorkflowState:
    """
    Reviewer Agent: A simple validator that checks if the drafted sections
    actually contain the citation markers provided by the retrieved chunks.
    """
    logger.info("ReviewerAgent: Validating drafted sections for citation markers")

    drafted_sections = state.get("drafted_sections", {})
    validation_results = {}

    try:
        for section_title, content in drafted_sections.items():
            # Check for citation markers in the format [Filename, Page: X] or similar patterns
            import re

            # Look for common citation patterns
            citation_patterns = [
                r'\[[^\]]+\]',  # General pattern for [something]
                r'\[Filename, Page: [^\]]+\]',  # Specific pattern from our format
                r'\[\d+\]',  # Numeric citations
                r'\([^)]+\d+[^)]*\)'  # Parenthetical citations with numbers
            ]

            has_citations = False
            for pattern in citation_patterns:
                if re.search(pattern, content):
                    has_citations = True
                    break

            validation_results[section_title] = {
                "has_citations": has_citations,
                "content_length": len(content),
                "status": "valid" if has_citations else "missing_citations"
            }

            if not has_citations:
                logger.warning(f"ReviewerAgent: Section '{section_title}' appears to be missing citation markers")

        # Overall status
        all_valid = all(result["has_citations"] for result in validation_results.values())

        return {
            **state,
            "current_status": "reviewed" if all_valid else "reviewed_with_warnings",
            # In a more complex implementation, we might store validation results in state
        }

    except Exception as e:
        logger.error(f"ReviewerAgent failed: {str(e)}")
        return {
            **state,
            "current_status": "review_failed"
        }

def create_research_workflow() -> StateGraph:
    """
    Creates and compiles the LangGraph workflow for the AI research process.
    """
    # Create the state graph
    workflow = StateGraph(WorkflowState)

    # Add nodes (we'll pass the dependencies via closure or config)
    workflow.add_node("planner", PlannerAgent)
    workflow.add_node("writer", WriterAgent)
    workflow.add_node("reviewer", ReviewerAgent)

    # Define the flow
    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", END)

    # Compile the workflow
    return workflow.compile()

def run_research_workflow(query: str, llm: LocalLLM = None, rag_engine: RAGEngine = None) -> Dict[str, Any]:
    """
    Main function to execute the research workflow.

    Args:
        query: The user's research prompt/query
        llm: Optional LocalLLM instance (will create default if not provided)
        rag_engine: Optional RAGEngine instance (will create default if not provided)

    Returns:
        Dictionary containing the final state and results
    """
    logger.info(f"Starting research workflow for query: {query[:100]}...")

    # Initialize dependencies if not provided
    if llm is None:
        llm = LocalLLM()

    if rag_engine is None:
        rag_engine = RAGEngine()

    # Create workflow
    app_workflow = create_research_workflow()

    # Initial state
    initial_state = WorkflowState(
        user_prompt=query,
        outline=[],
        drafted_sections={},
        current_status="initialized"
    )

    try:
        # Run the workflow
        # Note: We need to handle the fact that our nodes expect llm and rag_engine parameters
        # For now, we'll create wrapper functions or modify the approach

        # Since LangGraph nodes typically don't take extra parameters beyond state,
        # we'll create partial functions or use a different approach

        # For simplicity in this implementation, let's create wrapper functions
        def planner_wrapper(state):
            return PlannerAgent(state, llm)

        def writer_wrapper(state):
            return WriterAgent(state, llm, rag_engine)

        # Reviewer doesn't need extra params

        # Recreate workflow with wrappers
        workflow = StateGraph(WorkflowState)
        workflow.add_node("planner", planner_wrapper)
        workflow.add_node("writer", writer_wrapper)
        workflow.add_node("reviewer", ReviewerAgent)

        workflow.set_entry_point("planner")
        workflow.add_edge("planner", "writer")
        workflow.add_edge("writer", "reviewer")
        workflow.add_edge("reviewer", END)

        compiled_workflow = workflow.compile()

        # Execute
        final_state = compiled_workflow.invoke(initial_state)

        logger.info(f"Research workflow completed with status: {final_state['current_status']}")

        return {
            "success": True,
            "state": final_state,
            "outline": final_state.get("outline", []),
            "drafted_sections": final_state.get("drafted_sections", {}),
            "status": final_state.get("current_status", "unknown")
        }

    except Exception as e:
        logger.error(f"Research workflow failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "state": initial_state,
            "outline": [],
            "drafted_sections": {},
            "status": "workflow_failed"
        }

# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing the workflow
    pass