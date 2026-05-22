"""
RAG Engine for the AI Knowledge Platform.
Handles semantic chunking, vector storage, hybrid search, and re-ranking.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi
import re

logger = logging.getLogger(__name__)

class RAGEngine:
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection_name: str = "knowledge_base",
        chunk_size: int = 200,
        chunk_overlap: int = 50,
        BM25_K1: float = 1.5,
        BM25_B: float = 0.75,
    ):
        """
        Initialize the RAG engine.
        :param embedding_model_name: Name of the sentence-transformers model to use.
        :param qdrant_host: Host for Qdrant server.
        :param qdrant_port: Port for Qdrant server.
        :param collection_name: Name of the Qdrant collection.
        :param chunk_size: Target chunk size in words.
        :param chunk_overlap: Overlap between chunks in words.
        :param BM25_K1: BM25 parameter k1.
        :param BM25_B: BM25 parameter b.
        """
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.qdrant_client = QdrantClient(host=qdrant_host, port=qdrant_port)
        self.collection_name = collection_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.BM25_K1 = BM25_K1
        self.BM25_B = BM25_B

        # In-memory storage for chunks (for BM25 indexing)
        self.chunks: List[Dict[str, Any]] = []  # each chunk: {"text": str, "metadata": dict}
        self.chunk_texts: List[str] = []        # list of chunk texts for BM25
        self.bm25_index: Optional[BM25Okapi] = None

        # Ensure the collection exists
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure the Qdrant collection exists with the correct vector configuration."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection '{self.collection_name}' already exists.")
        except Exception:
            # Collection does not exist, create it
            vector_size = self.embedding_model.get_sentence_embedding_dimension()
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logger.info(f"Collection '{self.collection_name}' created with vector size {vector_size}.")

    def _chunk_text(self, text: str) -> List[str]:
        """
        Split text into chunks of approximately chunk_size words with overlap.
        :param text: Input text to chunk.
        :return: List of text chunks.
        """
        # Simple word-based chunking
        words = text.split()
        chunks = []
        start = 0
        while start < len(words):
            end = start + self.chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunks.append(chunk_text)
            start += self.chunk_size - self.chunk_overlap
        return chunks

    def add_chunks(self, chunks: List[Dict[str, Any]]) -> None:
        """
        Add pre-processed chunks to the knowledge base.
        :param chunks: List of dictionaries, each containing:
                       - "text": the chunk text
                       - "metadata": a dictionary with at least "doc_id", and optionally "page_number", "heading_path", etc.
        """
        # Prepare points for Qdrant and chunks for BM25
        points = []
        start_idx = len(self.chunks)  # starting index for new chunks

        for i, chunk in enumerate(chunks):
            chunk_text = chunk["text"]
            chunk_metadata = chunk["metadata"].copy()
            chunk_metadata["chunk_index"] = i
            chunk_data = {
                "text": chunk_text,
                "metadata": chunk_metadata
            }
            self.chunks.append(chunk_data)
            self.chunk_texts.append(chunk_text)

            # Generate embedding
            embedding = self.embedding_model.encode(chunk_text)

            # Prepare point for Qdrant
            point_id = start_idx + i
            points.append(PointStruct(
                id=point_id,
                vector=embedding.tolist(),
                payload=chunk_data
            ))

        # Upsert points to Qdrant
        if points:
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Added {len(points)} chunks to Qdrant.")

        # Rebuild BM25 index with all chunks
        self._rebuild_bm25_index()

    def _rebuild_bm25_index(self):
        """Rebuild the BM25 index from in-memory chunk texts."""
        if not self.chunk_texts:
            self.bm25_index = None
            return

        # Tokenize: simple lowercase and split by non-alphanumeric
        tokenized_corpus = [
            re.findall(r'\w+', text.lower()) for text in self.chunk_texts
        ]
        self.bm25_index = BM25Okapi(
            tokenized_corpus,
            k1=self.BM25_K1,
            b=self.BM25_B
        )
        logger.debug(f"Rebuilt BM25 index with {len(self.chunk_texts)} chunks.")

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        semantic_weight: float = 0.5,
        bm25_weight: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Query the knowledge base using hybrid search (semantic + BM25) with re-ranking.
        :param query_text: The query text.
        :param top_k: Number of results to return.
        :param semantic_weight: Weight for semantic search score (0-1).
        :param bm25_weight: Weight for BM25 search score (0-1).
        :return: List of results, each containing chunk text, metadata, and score.
        """
        if not self.chunks:
            logger.warning("No documents in the knowledge base.")
            return []

        # 1. Semantic search via Qdrant
        query_embedding = self.embedding_model.encode(query_text)
        semantic_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding.tolist(),
            limit=top_k * 2  # Get more candidates for re-ranking
        )

        # Extract semantic scores and chunk IDs
        semantic_scores = {}
        for point in semantic_results:
            chunk_id = point.id
            semantic_scores[chunk_id] = point.score  # cosine similarity

        # 2. BM25 search
        bm25_scores = {}
        if self.bm25_index and self.chunk_texts:
            # Tokenize query
            tokenized_query = re.findall(r'\w+', query_text.lower())
            # Get BM25 scores for all chunks
            all_bm25_scores = self.bm25_index.get_scores(tokenized_query)
            for chunk_id, score in enumerate(all_bm25_scores):
                bm25_scores[chunk_id] = score

        # 3. Combine scores and re-rank
        all_chunk_ids = set(semantic_scores.keys()) | set(bm25_scores.keys())
        combined_scores = {}

        # Normalize scores to [0, 1] for each method
        def normalize_scores(scores_dict):
            if not scores_dict:
                return {}
            values = list(scores_dict.values())
            min_val = min(values)
            max_val = max(values)
            if max_val == min_val:
                return {k: 0.5 for k in scores_dict}  # Avoid division by zero
            return {
                k: (v - min_val) / (max_val - min_val)
                for k, v in scores_dict.items()
            }

        norm_semantic = normalize_scores(semantic_scores)
        norm_bm25 = normalize_scores(bm25_scores)

        for chunk_id in all_chunk_ids:
            sem_score = norm_semantic.get(chunk_id, 0.0)
            bm25_score = norm_bm25.get(chunk_id, 0.0)
            combined = (semantic_weight * sem_score) + (bm25_weight * bm25_score)
            combined_scores[chunk_id] = combined

        # Sort by combined score descending and take top_k
        sorted_chunk_ids = sorted(
            combined_scores.keys(),
            key=lambda cid: combined_scores[cid],
            reverse=True
        )[:top_k]

        # 4. Retrieve chunk data for top chunk IDs
        results = []
        for chunk_id in sorted_chunk_ids:
            if chunk_id < len(self.chunks):
                chunk_data = self.chunks[chunk_id]
                results.append({
                    "text": chunk_data["text"],
                    "metadata": chunk_data["metadata"],
                    "score": combined_scores[chunk_id]
                })

        logger.info(f"Query returned {len(results)} results.")
        return results

# Example usage (for testing)
if __name__ == "__main__":
    # This is just for demonstration.
    engine = RAGEngine()
    # Assume we have processed a document and got its text and metadata
    # sample_text = "This is a sample document about artificial intelligence."
    # sample_metadata = {"filename": "sample.pdf", "page": 1}
    # engine.add_document(sample_text, sample_metadata)
    # results = engine.query("What is AI?", top_k=3)
    # for res in results:
    #     print(f"Text: {res['text'][:100]}...")
    #     print(f"Metadata: {res['metadata']}")
    #     print(f"Score: {res['score']}\n")
    pass