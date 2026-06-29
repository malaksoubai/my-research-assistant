# Confirm that all libraries are installed and working
# Expected output: 6 lines of OK responses

# PyMuPDF
import fitz
print(f"PyMuPDF OK - version {fitz.version}")

# 2. spaCy
import spacy
nlp = spacy.load("en_core_web_sm")
doc = nlp("The transformer model was proposed in 2017.")
print(f"spaCy OK - entities found: {[(ent.text, ent.label_) for ent in doc.ents]}")

# 3. Sentence Transformers
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("hello world")
print(f"Sentence Transformers OK - embedding shape: {embedding.shape}")

# 4. ChromaDB
import chromadb
client = chromadb.Client()
collection = client.create_collection("smoke_test")
print(f"ChromaDB OK - collection created: {collection.name}")

# 5. LlamaIndex core
import llama_index.core
print(f"LlamaIndex core OK - version {llama_index.core.__version__}")

# 6. Ollama (checks if the service is running)
# Ollama will likely show the "NOT running" message the first time. 
# That's expected — Ollama needs to be running as a background service separately from your terminal. 
# Open a second terminal and run ollama serve, then run the smoke test again
import requests
try:
    response = requests.get("http://localhost:11434")
    print(f"Ollama OK - service is running")
except requests.exceptions.ConnectionError:
    print("Ollama NOT running. Open a separate terminal and run: ollama serve")