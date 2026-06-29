# The goals of this script:
# 1. Read all PDFs from a folder and validate them
# 2. Extract text per page, chunk it, and attach metadata (filename, page_num, chunk_id)

import os
from pathlib import Path
import fitz
import config

def extract_uploads(folder: str) -> list[str]:
    """Extracts uploaded files under the data folder."""
    files = []
    folder_path = Path(folder)
    if folder_path.is_dir():
        for file in folder_path.iterdir():
            if file.is_file():
                files.append(str(file))
            else:
                print(f"Error: '{str(file)}' is not a valid file.")
    else:
        print(f"Error: '{folder}' is not a valid directory.")
    
    return files


def is_file_valid(file: str) -> tuple[bool, str]:
    """Checks if a file is PDF, non-empty, and openable."""
    if not file.lower().endswith(".pdf"):
        return False, f"{file} is not a .pdf file."
    if os.path.getsize(file) == 0:
        return False, f"{file} is an empty file."
    try:
        doc = fitz.open(file)
    except Exception as e:
        return False, f"Error opening {file}: {e}"
    
    doc.close()
    return True,  f"{file} is a valid .pdf file."


my_sources = extract_uploads(config.PDF_FOLDER)

for source in my_sources:
    print(is_file_valid(source)[1])



