# The goals of this script:
# DOCUMENT INGESTION
# 1. Read all PDFs from a folder and validate them
# 2. Extract text per page and clean it
# 3. Chunk text and attach metadata (filename, page_num, chunk_id)
# NLP PREPROCESSING
# 4. spaCy named entity extraction per chunk


import os
from pathlib import Path
import re
import fitz     # PyMuPDF
import config
import spacy

# --------------------------------------------------
# DOCUMENT INGESTION
# --------------------------------------------------
# 1. Read all PDFs from a folder and validate them using PyMuPDF
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

    print(f"    [STATUS:SUCCESS]  DATA_EXTRACTION:'{Path(file_path).name}'.")
    return pages

# --------------------------------------------------
# 3. Chunk text and attach metadata
# --------------------------------------------------

def chunk_text(pages: list[dict], file_path: str) -> list[dict]:
    """Chunks each page into overlapping-based chunks.
    Each chunk stores its respective metadata (filename, page, chunk_id)"""

    filename = Path(file_path).name

    print(f"Chunking {filename} staring now.")

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

    print(f"    [STATUS:SUCCESS]  DATA_CHUNKING:'{filename}'.")
    return chunks

# --------------------------------------------------
# NLP PREPROCESSING
# --------------------------------------------------
# 4. spaCy named entity extraction per chunk
# --------------------------------------------------

def extract_entities(text: str) -> str:
    """Cleans and annotates text using spaCy."""
    # tokenization
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    # NER
    ner = []
    for ent in doc.ents:
        entity = ent.text
        label = ent.label_
        # print(f"Entity: {entity:<12} | Label: {label:<12} | Explanation: {spacy.explain(label)}")
        ner.append(f"{entity} {label}")
    
    print(f"    [STATUS:SUCCESS]  ENTITY_EXTRACTION.")
    return ", ".join(ner) if ner else ""

# --------------------------------------------------
# Smoke test
# --------------------------------------------------


### smoke test 1-3: text extraction and chunking
# my_sources = extract_uploads(config.PDF_FOLDER)
# print(my_sources)

# for source in my_sources:
#     if is_file_valid(source)[0]:
#         extracted_pages = extract_pages(file_path=source)
#         # print(extracted_pages)
#         chunk_text(extracted_pages, source)

# print("D    O   N   E")

### smoke test 4: named entity recognition
test_sentence = "Apple is looking at buying U.K. startups for $1 billion. Employees loved working there."
print(extract_entities(text=test_sentence))