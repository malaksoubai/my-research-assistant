# Shared settings

# Paths
PDF_FOLDER = "data/pdfs"

# Min text to validate digital page
MIN_TEXT_PER_PAGE = 50

# Chunks
CHUNK_SIZE = 400
CHUNK_OVERLAP = 40

# spaCy NLP
NLP_MODEL = "en_core_web_sm"

# ChromaDB
P_DB_PATH = "./chroma_db"
COLLECTION_NAME = "research_papers"

# Sentence Transformer
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Ollama Model
# LLM_MODEL = "llama3.2:1b"

# Groq 
import os
from dotenv import load_dotenv

load_dotenv() # Load .env file
API_KEY = os.getenv("GROQ_API_KEY")
if not API_KEY: # Optional safeguard to catch missing keys early
    raise ValueError("Error: GROQ_API_KEY is not set in the .env file!")

LLM_MODEL = "llama-3.3-70b-versatile"