"""
Test file to verify the workflow structure without external dependencies.
This tests the LangGraph workflow structure and basic functionality.
"""

from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

class WorkflowState(TypedDict):
    """State for the AI writing workflow."""
    user_prompt: str
    outline: List[str]
    drafted_sections: Dict[str, str]
    current_status: str

def mock_planner_agent(state: WorkflowState) -> WorkflowState:
    """Mock planner agent for testing."""
    print("MockPlannerAgent: Generating outline")
    return {
        **state,
        "outline": ["Introduction", "Methodology", "Results", "Conclusion"],
        "current_status": "outline_generated"
    }

def mock_writer_agent(state: WorkflowState) -> WorkflowState:
    """Mock writer agent for testing."""
    print("MockWriterAgent: Drafting sections")
    drafted_sections = {}
    outline = state.get("outline", [])

    for section in outline:
        drafted_sections[section] = f"Content for {section} with [citation] markers."

    return {
        **state,
        "drafted_sections": drafted_sections,
        "current_status": "sections_drafted"
    }

def mock_reviewer_agent(state: WorkflowState) -> WorkflowState:
    """Mock reviewer agent for testing."""
    print("MockReviewerAgent: Reviewing sections")
    # Simple validation - check if content has citation markers
    drafted_sections = state.get("drafted_sections", {})
    all_have_citations = all("[citation]" in content for content in drafted_sections.values())

    return {
        **state,
        "current_status": "reviewed" if all_have_citations else "reviewed_with_warnings"
    }

def create_test_workflow() -> StateGraph:
    """Create a test workflow with mock agents."""
    workflow = StateGraph(WorkflowState)

    workflow.add_node("planner", mock_planner_agent)
    workflow.add_node("writer", mock_writer_agent)
    workflow.add_node("reviewer", mock_reviewer_agent)

    workflow.set_entry_point("planner")
    workflow.add_edge("planner", "writer")
    workflow.add_edge("writer", "reviewer")
    workflow.add_edge("reviewer", END)

    return workflow.compile()

def test_workflow():
    """Test the workflow execution."""
    print("Testing AI Writing Workflow...")

    # Create workflow
    app = create_test_workflow()

    # Initial state
    initial_state = WorkflowState(
        user_prompt="Test query about artificial intelligence",
        outline=[],
        drafted_sections={},
        current_status="initialized"
    )

    # Run workflow
    final_state = app.invoke(initial_state)

    print(f"Final status: {final_state['current_status']}")
    print(f"Outline: {final_state['outline']}")
    print(f"Number of drafted sections: {len(final_state['drafted_sections'])}")

    for section, content in final_state['drafted_sections'].items():
        print(f"- {section}: {content[:50]}...")

    assert final_state['current_status'] == 'reviewed'
    assert len(final_state['outline']) == 4
    assert len(final_state['drafted_sections']) == 4

    print("✅ Workflow test passed!")
    return True

if __name__ == "__main__":
    test_workflow()