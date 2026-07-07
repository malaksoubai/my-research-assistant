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
E_DB_PATH = "./ephemeral_db"
COLLECTION_NAME = ""

# Sentence Transformer
EMBEDDING_MODEL = "all-MiniLM-L6-v2"