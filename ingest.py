# The goals of this script:
# DOCUMENT INGESTION
# 1. Read all PDFs from a folder and validate them
# 2. Extract text per page and clean it
# 3. Chunk text and attach metadata (filename, page_num, chunk_id)
# NLP PREPROCESSING
# 4. spaCy named entity extraction per chunk
# EMBEDDING + STORAGE
# 5. vectorize chunks and store in chromaDB


import config
import os

from pathlib import Path
import re
import fitz     # PyMuPDF
import spacy
import chromadb
from sentence_transformers import SentenceTransformer
from llama_index.llms.ollama import Ollama
from llama_index.llms.groq import Groq

# --------------------------------------------------
# SETUP
# --------------------------------------------------
# 0. Load all tools
# --------------------------------------------------

def load_tools():
    """Load all necessary tools once to prevent reload."""
    print(f"    [STATUS:STARTED]  TOOLS LOADING.")
    try:
        # tokenization
        nlp = spacy.load(config.NLP_MODEL)

        # save data in a local directory, db instance
        client = chromadb.PersistentClient(path=config.P_DB_PATH)

        # vectorization (list of nums) for embeddings, rows in table 
        embedder = SentenceTransformer(config.EMBEDDING_MODEL)

        # table 
        # NOTE: use following two lines only to wipe all entries of the collection
        # client.delete_collection(config.COLLECTION_NAME)
        # print("Vector database wiped and recreated.")

        collection = client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata = {"hnsw:space": "cosine"},    # Hierarchical Navigable Small World index
        )

        # When using Ollama:
        # llm = Ollama(model=config.LLM_MODEL, request_timeout=120.0)
        # When using Groq:
        llm = Groq(model=config.LLM_MODEL, api_key=config.API_KEY)
        
    except Exception as e:
        print(f"    [STATUS:FAILED]  TOOLS NOT LOADED.")
        print(f"ERROR occurred: {e}")
        return

    print(f"    [STATUS:SUCCESS]  TOOLS LOADED.")

    return nlp, embedder, collection, llm

# --------------------------------------------------
# DOCUMENT INGESTION
# --------------------------------------------------
# 1. Read all PDFs and validate them using PyMuPDF
# --------------------------------------------------

def extract_uploads(folder: str) -> list[str]:
    """Extracts uploaded files under the data folder and returns their paths."""
    files = []
    folder_path = Path(folder)
    if folder_path.is_dir():
        for file_path in folder_path.iterdir():
            if file_path.is_file():
                files.append(str(file_path))
            else:
                print(f"Error: '{str(file_path)}' is not a valid file.")
    else:
        print(f"Error: '{folder}' is not a valid directory.")
    
    return files


def is_file_valid(file_path: str) -> tuple[bool, str]:
    """Checks if a file is PDF, non-empty, and openable."""
    if not file_path.lower().endswith(".pdf"):
        return False, f"{file_path} is not a .pdf file."
    if os.path.getsize(file_path) == 0:
        return False, f"{file_path} is an empty file."
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        return False, f"Error opening {file_path}: {e}"
    
    doc.close()
    return True,  f"{file_path} is a valid .pdf file."

def is_already_ingested(filename: str, collection) -> bool:
    """Checks if a file has already been ingested."""
    # collection.get() method filters by metadata
    result = collection.get(where={"filename":filename}, limit = 1)
    return len(result["ids"]) > 0

# --------------------------------------------------
# 2. Extract text per page and clean it
# --------------------------------------------------

def clean_text(text: str) -> str:
    """Cleans text from common PDF artifacts:
    - ligature characters (ﬁ, ﬂ, etc.) that break word matching
    - arXiv watermark lines, e.g. "arXiv:1708.08197v1  [cs.CV]  28 Aug 2017"
    - excessive whitespace/newlines from PDF layout wrapping
    """

    # fix ligature
    ligature_map = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    for broken, fixed in ligature_map.items():
        text = text.replace(broken, fixed)

    # remove arXiv
    text = re.sub(r"arXiv:\S+\s+\[\S+\]\s+\d{1,2}\s\w+\s\d{4}", "", text)

    # collapse newlines/spaces into single space
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_pages(file_path: str) -> list[dict]:
    """Returns  a list of {page_num: text} for every page in file.
    Flags pages with suspiciously little text which were likely scanned and not digital."""
    try:
        doc = fitz.open(file_path)
    except Exception as e:
        print(f"Error occurred while opening {file_path}: {e}")
        return []
    
    pages = []
    for page_num, page in enumerate(doc, start = 1):
        text = clean_text(page.get_text())

        if len(text) < config.MIN_TEXT_PER_PAGE:
            print(f"    WARNING: page {page_num} has very little text ({len(text)} length).")
        
        pages.append({"page_num": page_num, "text": text})

    doc.close()

    print(f"    [STATUS:SUCCESS]  DATA EXTRACTED:'{Path(file_path).name}'.")
    return pages

# --------------------------------------------------
# 3. Chunk text and attach metadata
# --------------------------------------------------

def chunk_text(pages: list[dict], filename: str) -> list[dict]:
    """Chunks each page into overlapping-based chunks.
    Each chunk stores its respective metadata (filename, page, chunk_id)"""

    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split() # type(words) = list[str]

        if not words:
            # skip empty pages
            continue

        step = config.CHUNK_SIZE - config.CHUNK_OVERLAP

        # step 1 = 400 - 40 = 360

        for i in range(0, len(words), step):
            chunk_list = words[i : i + config.CHUNK_SIZE]

            if chunk_list:
                # if chunk is not empty:
                chunk_str = " ".join(chunk_list)

                chunks.append({
                    "text": chunk_str,
                    "filename": filename,
                    "page_num": page["page_num"],
                    "chunk_id": f"{filename}_{chunk_id}"
                })

                chunk_id +=1 

    print(f"    [STATUS:SUCCESS]  DATA CHUNKED.")
    return chunks

# --------------------------------------------------
# NLP PREPROCESSING
# --------------------------------------------------
# 4. spaCy named entity extraction per chunk
# --------------------------------------------------

def extract_entities(text: str, nlp) -> str:
    """Cleans and annotates text using spaCy."""
    # now used in load_tools(): nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # NER - needed for semantic search/retrieval
    ner = []
    for ent in doc.ents:
        entity = ent.text
        label = ent.label_
        # print(f"Entity: {entity:<12} | Label: {label:<12} | Explanation: {spacy.explain(label)}")
        ner.append(f"{entity} {label}")
    
    # print(f"    [STATUS:SUCCESS]  ENTITY EXTRACTED.")
    return ", ".join(ner) if ner else ""

# --------------------------------------------------
# EMBEDDING + STORAGE
# --------------------------------------------------
# 5. vectorize chunks and store in chromaDB
# --------------------------------------------------

def embed_and_store(chunks: list[dict], nlp, embedder, collection) -> None:
    """Vectorize chunks and store them in chromaDB."""
    if not chunks:
        return

    texts = [c["text"] for c in chunks]
    ids = [c["chunk_id"] for c in chunks] # ids=filename_chunkId

    try:
        embeddings = embedder.encode(texts).tolist()
    except Exception as e:
        print(f"ERROR occurred: {e}")
        print(f"    [STATUS:FAILED]  VECTOR NOT EMBEDDED.")
        return
    
    print(f"    [STATUS:SUCCESS]  VECTOR EMBEDDED.")

    metadatas = []

    for c in chunks:
        entities = extract_entities(c["text"], nlp)
        metadatas.append({
            "filename": c["filename"],
            "page": c["page_num"],
            "entities": entities,
        })

    try:
        collection.add(    
            ids = ids,                  # required (list[str])
            embeddings = embeddings,    # optional (list[list[float]])
            documents = texts,          # optional (list[str])
            metadatas = metadatas       # optional (list[dict])
        )

    except Exception as e:
        print(f"ERROR occurred: {e}")
        print(f"    [STATUS:FAILED]  VECTOR NOT STORED.")
        return

    print(f"    [STATUS:SUCCESS]  VECTOR STORED.")


# --------------------------------------------------
# 6. Ingestion pipeline
# --------------------------------------------------

def ingest(file_path: str, nlp, embedder, collection) -> dict:
    """Runs the full ingestion pipeline for one PDF.
    Returns a summary used in the final report."""  
    
    filename = Path(file_path).name
    valid, reason = is_file_valid(file_path=file_path)
    
    if not valid:
        print(f"{filename} skipped: {reason}")
        return {"filename": filename, "status": "skipped", "reason": reason, "chunks": 0}
    
    if is_already_ingested(filename, collection):
        print(f"{filename} skipped: File already ingested.")
        return {"filename": filename, "status": "skipped", "reason": "File already ingested.", "chunks": 0}

    pages = extract_pages(file_path=file_path)
    chunk_list = chunk_text(pages=pages, filename=filename)
    embed_and_store(chunks=chunk_list, nlp=nlp, embedder=embedder, collection=collection)
    
    return {"filename": filename, "status": "ingested", "reason": "", "chunks": len(chunk_list)}

# --------------------------------------------------
# 7. Main function
# --------------------------------------------------

def main() -> None:
    """Runs the ingestion pipeline on all files once and stores embedding in db."""
    nlp, embedder, collection, llm = load_tools()

    sources = extract_uploads(config.PDF_FOLDER)

    if not sources:
        print(f"No source was found under {config.PDF_FOLDER}")
        return
    
    print(f"Sources uploaded: {sources}")

    ingested = []
    skipped = []
    new_chunks = 0
    for source in sources:
        result = ingest(source, nlp, embedder, collection)
        if result["status"] == "ingested":
            ingested.append(result)
            new_chunks += result["chunks"]
        else:
            skipped.append(result)

        
    # summary report
    print("-" * 50)
    print("INGESTION SUMMARY")
    print("-" * 50)
    print(f"PDFs found: {len(sources)}")
    print(f"Ingested: {len(ingested)}")
    print(f"Skipped: {len(skipped)}")
    print(f"New Chunks: {new_chunks}")
    print(f"Collection size: {collection.count()} chunks in ChromaDB.")


if __name__ == "__main__":
    main()


# --------------------------------------------------
# SMOKE TESTS
# --------------------------------------------------
# Add smoke test for sanity checks here

# client.list_collections() returns up to 100 collections


