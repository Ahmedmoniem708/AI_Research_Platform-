"""
Configuration settings for the AI Knowledge Platform.
Loads environment variables and provides default values.
"""

import os
from typing import List, Union
from pydantic import BaseSettings, Field

class Settings(BaseSettings):
    # API Settings
    PROJECT_NAME: str = "AI Knowledge Platform"
    VERSION: str = "0.1.0"
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False

    # CORS Settings
    BACKEND_CORS_ORIGINS: List[str] = ["*"]

    # OCR Settings
    DEFAULT_OCR_LANG: str = "en"  # Options: 'en', 'ch', 'fr', 'german', 'korean', 'japan', 'arabic'

    # RAG Settings
    EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "knowledge_base"
    CHUNK_SIZE: int = 200  # words
    CHUNK_OVERLAP: int = 50  # words

    # LLM Settings
    LLM_MODEL: str = "microsoft/phi-2"  # Or any other model supported by vLLM
    LLM_TENSOR_PARALLEL_SIZE: int = 1
    LLM_MAX_MODEL_LEN: int = 2048
    LLM_DTYPE: str = "auto"  # Options: "auto", "float16", "bfloat16", etc.

    class Config:
        env_file = ".env"
        case_sensitive = True

# Create global settings instance
settings = Settings()