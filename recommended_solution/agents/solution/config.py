import os
from dotenv import load_dotenv

load_dotenv()

# Default model for Hugging Face Inference
LLM_MODEL = os.getenv("LLM_MODEL", "Qwen/Qwen3.5-9B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")

# Qdrant Vector DB Configuration
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "telecom_knowledge_base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
