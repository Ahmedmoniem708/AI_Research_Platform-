"""
Generation module for the AI Knowledge Platform.
Integrates with local LLM via vLLM and handles citation formatting.
"""

from typing import List, Dict, Any, Optional
import logging
from vllm import LLM, SamplingParams

logger = logging.getLogger(__name__)

class LocalLLM:
    def __init__(
        self,
        model_name: str = "microsoft/phi-2",
        tensor_parallel_size: int = 1,
        max_model_len: int = 2048,
        dtype: str = "auto",
        trust_remote_code: bool = True,
    ):
        """
        Initialize the local LLM using vLLM.
        :param model_name: Name or path of the model to load.
        :param tensor_parallel_size: Number of GPUs to use for tensor parallelism.
        :param max_model_len: Maximum sequence length for the model.
        :param dtype: Data type for model weights ("auto", "float16", "bfloat16", etc.).
        :param trust_remote_code: Whether to trust remote code when loading the model.
        """
        self.model_name = model_name
        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=max_model_len,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
        )

        # Default sampling parameters
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.9,
            max_tokens=512,
            stop_token_ids=[self.llm.get_tokenizer().eos_token_id]
        )

        logger.info(f"Initialized vLLM with model: {model_name}")

    def generate_with_citations(
        self,
        query: str,
        retrieved_chunks: List[Dict[str, Any]],
        citation_format: str = "[Filename, Page: {page}]",
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Dict[str, Any]:
        """
        Generate an answer based on retrieved chunks with proper citations.
        :param query: The user's query.
        :param retrieved_chunks: List of retrieved chunks from RAG engine.
        :param citation_format: Format string for citations (should contain {filename} and {page} placeholders).
        :param max_new_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature.
        :param top_p: Top-p sampling parameter.
        :return: Dictionary containing the generated answer and metadata.
        """
        if not retrieved_chunks:
            return {
                "answer": "I don't have enough information to answer that question.",
                "citations": [],
                "prompt_used": "",
                "usage": {}
            }

        # Prepare context from retrieved chunks
        context_parts = []
        citation_map = {}  # To track which chunks contribute to which citations

        for i, chunk in enumerate(retrieved_chunks):
            metadata = chunk["metadata"]
            # Try to get the pre-built citation_tag, otherwise fall back to formatting
            citation = metadata.get("citation_tag")
            if citation is None:
                filename = metadata.get("filename", "Unknown")
                page = metadata.get("page", "Unknown")
                citation = citation_format.format(filename=filename, page=page)

            # Add to context with citation marker
            context_parts.append(f"[{citation}] {chunk['text']}")

            # Store for later reference
            citation_map[i] = {
                "citation": citation
            }

        # Join context parts
        context = "\n\n".join(context_parts)

        # Construct the prompt with strict instruction for citations
        prompt = f"""You are an AI assistant that answers questions based on provided context.
You MUST answer the question using ONLY the information provided in the context.
For every fact or statement you make, you MUST provide a citation in the exact format: [Filename, Page: X]
If you cannot answer the question based on the context, say you don't have enough information.

Context:
{context}

Question: {query}

Answer:"""

        # Update sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_new_tokens,
            stop_token_ids=[self.llm.get_tokenizer().eos_token_id]
        )

        # Generate response
        outputs = self.llm.generate([prompt], sampling_params)
        generated_text = outputs[0].outputs[0].text.strip()

        # Extract citations from the generated text (simple approach: look for citation patterns)
        # In a more sophisticated implementation, you might track which citations were actually used
        citations_used = list(set([chunk["citation"] for chunk in citation_map.values()]))

        return {
            "answer": generated_text,
            "citations": citations_used,
            "prompt_used": prompt,
            "usage": {
                "prompt_tokens": len(self.llm.get_tokenizer().encode(prompt)),
                "generated_tokens": len(self.llm.get_tokenizer().encode(generated_text))
            }
        }

# Example usage (for testing)
if __name__ == "__main__":
    # This is just for demonstration.
    # llm = LocalLLM(model_name="microsoft/phi-2")
    # sample_chunks = [
    #     {
    #         "text": "Artificial intelligence (AI) is intelligence demonstrated by machines.",
    #         "metadata": {"filename": "AI_Intro.pdf", "page": 1}
    #     },
    #     {
    #         "text": "Machine learning is a subset of AI that focuses on algorithms that learn from data.",
    #         "metadata": {"filename": "ML_Basics.pdf", "page": 2}
    #     }
    # ]
    # result = llm.generate_with_citations("What is AI?", sample_chunks)
    # print(f"Answer: {result['answer']}")
    # print(f"Citations: {result['citations']}")
    pass