"""
Pydantic schemas for the AI Knowledge Platform API.
Defines request and response models.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    services: Dict[str, str] = Field(..., description="Status of individual services")

class DocumentUploadResponse(BaseModel):
    """Response model for document upload."""
    filename: str = Field(..., description="Name of the uploaded file")
    status: str = Field(..., description="Processing status")
    text_extracted: bool = Field(..., description="Whether text was extracted")
    text_length: int = Field(..., description="Length of extracted text in characters")
    chunks_created: int = Field(..., description="Number of text chunks created")
    metadata: Dict[str, Any] = Field(..., description="Metadata associated with the document")

class QueryRequest(BaseModel):
    """Request model for querying the knowledge base."""
    question: str = Field(..., description="The question to ask")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results to retrieve")
    semantic_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for semantic search")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Weight for BM25 search")
    max_new_tokens: int = Field(default=512, ge=1, le=2048, description="Maximum tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(default=0.9, ge=0.0, le=1.0, description="Top-p sampling parameter")

class QueryResponse(BaseModel):
    """Response model for querying the knowledge base."""
    answer: str = Field(..., description="Generated answer with citations")
    citations: List[str] = Field(..., description="List of citations used")
    retrieved_chunks: List[Dict[str, Any]] = Field(..., description="Retrieved chunks with scores")
    usage: Dict[str, int] = Field(..., description="Token usage statistics")