import os
from dotenv import load_dotenv
load_dotenv()

BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_GEN_PATH = os.getenv("OLLAMA_GEN_PATH", "/api/generate")  
OLLAMA_EMB_PATH = os.getenv("OLLAMA_EMB_PATH", "/api/embed")  
OLLAMA_MODEL_GEN = os.getenv("OLLAMA_MODEL_GEN", "qwen2.5:14b")
OLLAMA_MODEL_EMB = os.getenv("OLLAMA_MODEL_EMB", "nomic-embed-text")

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379")

FAISS_INDEX_PATH = os.getenv("FAISS_INDEX_PATH", "faiss.index")
METADATA_DB = os.getenv("METADATA_DB", "metadata.sqlite")
KAG_DB = os.getenv("KAG_DB", "kag.sqlite")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "400"))
CACHE_TTL = int(os.getenv("CACHE_TTL", "3600"))