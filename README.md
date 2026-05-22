# AI Knowledge Platform - RAG Proof of Concept

A professional, modular Python implementation of an AI Knowledge Platform using Retrieval-Augmented Generation (RAG) to search within documents and answer questions with citations.

## Features

- **Document Processing Pipeline**: OpenCV for image preprocessing (denoising, binarization) and PaddleOCR for high-accuracy text extraction (supports Arabic & English)
- **RAG Architecture**: 
  - Semantic chunking with metadata preservation (filename, page number)
  - Vector store implementation using Qdrant
  - Hybrid search functionality (Semantic + Keyword/BM25)
  - Re-ranking step to improve retrieval accuracy
- **Generation & Citations**: 
  - Integration with local LLM via vLLM
  - Strict prompt engineering to force model to output citations in format: `[Filename, Page: X]`
- **Backend**: FastAPI framework exposing endpoints for file upload, indexing, and querying

## Project Structure

```
AI_Research_Platform/
├── app/
│   ├── __init__.py
│   ├── main.py                 # Application entry point
│   ├──
│   ├── processing/             # Document processing pipeline
│   │   ├── __init__.py
│   │   └── processing.py       # OpenCV preprocessing + PaddleOCR
│   │
│   ├── rag/                    # RAG architecture
│   │   ├── __init__.py
│   │   └── engine.py           # Semantic chunking, Qdrant, hybrid search, re-ranking
│   │
│   ├── generation/             # LLM integration
│   │   ├── __init__.py
│   │   └── generation.py       # vLLM integration with citation formatting
│   │
│   ├── api/                    # API endpoints
│   │   ├── __init__.py
│   │   └── routes.py           # File upload, indexing, querying endpoints
│   │
│   ├── schemas/                # Pydantic models
│   │   ├── __init__.py
│   │   └── schemas.py          # Request/response models
│   │
│   ├── core/                   # Configuration
│   │   ├── __init__.py
│   │   └── config.py           # Settings management
│   │
│   ├── utils/                  # Utility functions (empty for now)
│   │   └── __init__.py
│   │
│   └── config/                 # Additional config (empty for now)
│       └── __init__.py
│
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Docker configuration
├── docker-compose.yml          # Docker Compose with Qdrant
├── .env                        # Environment variables
└── README.md                   # This file
```

## Installation

### Option 1: Local Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd AI_Research_Platform
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Start Qdrant (required for vector storage):
```bash
# Using Docker
docker run -p 6333:6333 -p 6334:6334 \
  -v $(pwd)/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:v1.6.3
```

4. Run the application:
```bash
uvicorn app.main:app --reload
```

### Option 2: Docker Installation

1. Start all services with Docker Compose:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## Usage

### API Endpoints

- **Health Check**: `GET /api/v1/health`
- **Upload Document**: `POST /api/v1/upload`
  - Parameters:
    - `file`: Image file (JPG, PNG, etc.)
    - `extract_text`: Boolean (default: true)
    - `metadata`: JSON string (optional)
- **Query Knowledge Base**: `POST /api/v1/query`
  - Parameters:
    - `question`: Your question
    - `top_k`: Number of results to retrieve (default: 5)
    - `semantic_weight`: Weight for semantic search (default: 0.5)
    - `bm25_weight`: Weight for BM25 search (default: 0.5)
    - `max_new_tokens`: Maximum tokens to generate (default: 512)
    - `temperature`: Sampling temperature (default: 0.7)
    - `top_p`: Top-p sampling parameter (default: 0.9)

### Example Usage with curl

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Upload a document
curl -X POST "http://localhost:8000/api/v1/upload" \
  -F "file=@document.jpg" \
  -F 'metadata={"filename": "document.jpg", "page": 1}'

# Query the knowledge base
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the main topic of the document?",
    "top_k": 5,
    "semantic_weight": 0.7,
    "bm25_weight": 0.3
  }'
```

## Technology Stack

- **Backend**: FastAPI, Uvicorn
- **Document Processing**: OpenCV, PaddleOCR
- **Embeddings**: Sentence Transformers
- **Vector Database**: Qdrant
- **Hybrid Search**: BM25 (rank_bm25)
- **LLM Integration**: vLLM
- **Configuration**: Pydantic Settings
- **Data Validation**: Pydantic
- **Containerization**: Docker, Docker Compose

## Configuration

Configuration can be managed through:
1. Environment variables (`.env` file)
2. Direct modification of `app/core/config.py`
3. Override via Docker Compose environment section

Key configuration options:
- `DEFAULT_OCR_LANG`: Language for OCR (`en`, `arabic`, etc.)
- `EMBEDDING_MODEL`: Sentence transformer model for embeddings
- `QDRANT_HOST/PORT`: Qdrant connection settings
- `CHUNK_SIZE/OVERLAP`: Text chunking parameters
- `LLM_MODEL`: Local LLM model via vLLM
- `LLM_TENSOR_PARALLEL_SIZE`: GPU tensor parallelism

## Notes for Production Use

1. **Security**: This is a PoC implementation. For production, add:
   - Authentication and authorization
   - Input validation and sanitization
   - Rate limiting
   - HTTPS termination

2. **Scalability**: 
   - For higher volume, consider using a distributed Qdrant deployment
   - Add caching layer (Redis) for frequent queries
   - Use GPU acceleration for OCR and LLM inference

3. **Monitoring**:
   - Add logging aggregation (ELK stack)
   - Implement metrics collection (Prometheus/Grafana)
   - Add health checks for all dependencies

4. **Document Processing Enhancements**:
   - Multi-page document support (PDF handling)
   - Layout analysis for complex documents
   - Language detection per document
   - Confidence scoring for OCR results

## Troubleshooting

1. **OCR Issues**: Ensure Tesseract dependencies are installed if using PaddleOCR with certain languages
2. **Qdrant Connection**: Verify Qdrant is running and accessible at the configured host/port
3. **GPU Issues**: If using GPU, ensure NVIDIA drivers and CUDA toolkit are properly installed
4. **Memory Issues**: Adjust chunk size and batch sizes based on available memory

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) for OCR capabilities
- [Sentence Transformers](https://www.sbert.net/) for embeddings
- [Qdrant](https://qdrant.tech/) for vector search
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inference
- [rank_bm25](https://github.com/dorianbrown/rank_bm25) for BM25 implementation