"""
API routes for the AI Knowledge Platform.
Defines endpoints for file upload, document indexing, and querying.
"""

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import numpy as np
import cv2
import logging
import tempfile
import os
from pydantic import BaseModel

from app.processing.processing import DocumentProcessor
from app.rag.engine import RAGEngine
from app.generation.generation import LocalLLM
from app.core.config import settings
from app.workflow.agents import run_research_workflow
from app.schemas.schemas import (
    QueryRequest,
    QueryResponse,
    DocumentUploadResponse,
    HealthCheckResponse
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic schema for research request
class ResearchRequest(BaseModel):
    query: str

# Initialize components (in a real app, these would be dependency injected)
doc_processor = DocumentProcessor(lang=settings.DEFAULT_OCR_LANG)
rag_engine = RAGEngine(
    embedding_model_name=settings.EMBEDDING_MODEL,
    qdrant_host=settings.QDRANT_HOST,
    qdrant_port=settings.QDRANT_PORT,
    collection_name=settings.QDRANT_COLLECTION_NAME,
    chunk_size=settings.CHUNK_SIZE,
    chunk_overlap=settings.CHUNK_OVERLAP
)
local_llm = LocalLLM(
    model_name=settings.LLM_MODEL,
    tensor_parallel_size=settings.LLM_TENSOR_PARALLEL_SIZE,
    max_model_len=settings.LLM_MAX_MODEL_LEN,
    dtype=settings.LLM_DTYPE
)


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """
    Health check endpoint.
    """
    return HealthCheckResponse(
        status="healthy",
        version=settings.VERSION,
        services={
            "api": "operational",
            "ocr": "operational",
            "rag": "operational" if rag_engine.qdrant_client else "unavailable",
            "llm": "operational" if local_llm.llm else "unavailable"
        }
    )


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    extract_text: bool = Form(True),
    metadata: str = Form("{}")
):
    """
    Upload and process a document.
    :param file: The document file (image format supported by OpenCV).
    :param extract_text: Whether to extract text using OCR.
    :param metadata: JSON string containing additional metadata.
    """
    try:
        # Parse metadata
        import json
        metadata_dict = json.loads(metadata) if metadata else {}

        # Add filename to metadata
        metadata_dict["filename"] = file.filename

        # Read file content
        contents = await file.read()

        # Convert to OpenCV image
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        # Process document if text extraction is requested
        if extract_text:
            # Process the document (image or PDF) to get chunks with metadata
            chunks = doc_processor.process_document(contents, file.filename)

            # Add the chunks to the RAG engine
            rag_engine.add_chunks(chunks)

            # Compute full text and text length for response
            full_text = " ".join([chunk["text"] for chunk in chunks])
            text_length = len(full_text)
            chunks_created = len(chunks)

            return DocumentUploadResponse(
                filename=file.filename,
                status="processed",
                text_extracted=True,
                text_length=text_length,
                chunks_created=chunks_created,
                metadata={}  # Metadata is now per-chunk, not per-document
            )
        else:
            # Just store metadata without processing
            return DocumentUploadResponse(
                filename=file.filename,
                status="uploaded",
                text_extracted=False,
                text_length=0,
                chunks_created=0,
                metadata=metadata_dict
            )

    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata JSON")
    except Exception as e:
        logger.error(f"Error processing document: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base and generate an answer with citations.
    :param request: Query request containing the question and parameters.
    """
    try:
        # Retrieve relevant chunks from RAG engine
        retrieved_chunks = rag_engine.query(
            query_text=request.question,
            top_k=request.top_k,
            semantic_weight=request.semantic_weight,
            bm25_weight=request.bm25_weight
        )

        if not retrieved_chunks:
            return QueryResponse(
                answer="I don't have enough information to answer that question.",
                citations=[],
                retrieved_chunks=[],
                usage={}
            )

        # Generate answer with citations using local LLM
        generation_result = local_llm.generate_with_citations(
            query=request.question,
            retrieved_chunks=retrieved_chunks,
            citation_format="[Filename, Page: {page}]",
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p
        )

        return QueryResponse(
            answer=generation_result["answer"],
            citations=generation_result["citations"],
            retrieved_chunks=[
                {
                    "text": chunk["text"][:200] + "..." if len(chunk["text"]) > 200 else chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": chunk["score"]
                }
                for chunk in retrieved_chunks
            ],
            usage=generation_result["usage"]
        )

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/documents", response_model=List[Dict[str, Any]])
async def list_documents():
    """
    List all documents in the knowledge base.
    """
    try:
        # Get collection info from Qdrant
        collection_info = rag_engine.qdrant_client.get_collection(
            collection_name=settings.QDRANT_COLLECTION_NAME
        )

        # Scroll through points to get unique documents
        # This is a simplified implementation - in production you'd want better document tracking
        points, _ = rag_engine.qdrant_client.scroll(
            collection_name=settings.QDRANT_COLLECTION_NAME,
            limit=1000,
            with_payload=True
        )

        # Extract unique documents based on filename
        documents = {}
        for point in points:
            payload = point.payload
            if payload and "metadata" in payload:
                metadata = payload["metadata"]
                filename = metadata.get("filename", "unknown")
                if filename not in documents:
                    documents[filename] = {
                        "filename": filename,
                        "pages": set(),
                        "chunks": 0
                    }
                if "page" in metadata:
                    documents[filename]["pages"].add(metadata["page"])
                documents[filename]["chunks"] += 1

        # Convert sets to lists for JSON serialization
        result = []
        for doc in documents.values():
            doc["pages"] = sorted(list(doc["pages"]))
            result.append(doc)

        return result

    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/generate-research")
async def generate_research(request: ResearchRequest):
    """
    Generate a research outline and drafted sections based on the user's query.
    :param request: Research request containing the query.
    """
    try:
        # Run the research workflow
        result = run_research_workflow(request.query)

        if not result["success"]:
            raise HTTPException(status_code=500, detail=f"Research workflow failed: {result['error']}")

        # Return the outline and drafted sections
        return {
            "outline": result["outline"],
            "drafted_sections": result["drafted_sections"],
            "status": result["status"]
        }

    except Exception as e:
        logger.error(f"Error generating research: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")