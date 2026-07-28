# Research Assistant
A RAG pipeline that ingests PDFs, retrieves relevant passages semantically, and answers questions with cited sources. Built as a learning project to learn more about NLP and AI engineering at the industry level.


![Status](https://img.shields.io/badge/Status-Complete-brightgreen)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Table of Content
1. Overview (what it does, who it's for)
2. Tech stack (tools used)
3. Architecture (how it works: diagram link or description)
4. Setup & installation (exact steps to run it)
5. Usage (how to actually use it)
6. Evaluation results (findings)
7. Limitations
8. What I learned (personal project)

## Overview
The Research Assistant is a full Retrieval-Augmented Generation (RAG) pipeline that ingests PDFs, retrieves relevant chunks semantically, and answers questions with cited sources pointing to the file name and page number used as source material. The project includes an evaluation dashboard to test different system settings and their retrieval accuracy score.

This may useful for anyone who reads more papers than they can remember, like students, analysts, researchers, etc. They can ask questions across a library of PDFs instead of ctrl-F-ing one at a time. It was mainly develop as a learning tool to gain hands-on experience with the core tools of NLP and AI engineering.

## Tech Stack 
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyMuPDF](https://img.shields.io/badge/PyMuPDF-PDF_Parsing-green)](https://pymupdf.readthedocs.io/en/latest/)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?logo=spacy&logoColor=white)](https://spacy.io/)
[![Sentence Transformers](https://img.shields.io/badge/Sentence_Transformers-Embeddings-orange)](https://sbert.net/)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6B35)](https://www.trychroma.com/products/chromadb)
[![Groq](https://img.shields.io/badge/Groq-LLM_API-F55036)](https://groq.com)
[![LLM](https://img.shields.io/badge/Llama_3.3_70B-Generation-8A2BE2)](https://console.groq.com/docs/model/llama-3.3-70b-versatile)

## Architecture
There are two main pipelines used in the project.

Ingestion (runs once per batch of PDFs, or after system level changes):

> PDF files → PyMuPDF extraction → text cleaning → chunking + metadata → spaCy NER → Sentence Transformers embeddings → ChromaDB

Query (runs on every user question):

> User question → embed query → ChromaDB similarity search → filter by threshold → build prompt → Groq LLM → answer + citations

## Setup

### Prerequisites

- Install [Python 3.11](https://www.python.org/)

- Get free [Groq API Key](console.groq.com). Create a .env file in the project root:
```bash
GROQ_API_KEY="your_key_here"
```

### Installation
```bash
# 1. clone the repo
git clone https://github.com/malaksoubai/my-research-assistant.git
cd my-research-assistant

# 2. create and activate virtual environment
python -m venv venv
source venv/Scripts/activate       # Windows (Git Bash)
# source venv/bin/activate         # Mac/Linux

# 3. install dependencies
pip install -r requirements.txt

# 4. download spaCy language model
python -m spacy download en_core_web_sm
```

## Usage

1. Add PDFs

Place PDF files in the `data/pdfs/` folder. 

2. Ingest

Make sure you are inside the virtual environnement.

```bash
# ingest new PDFs (skips already-ingested files and only ingests new ones)
python ingest.py

# wipe collection and re-ingest from scratch (e.g. after changing chunk size)
python ingest.py --reset
```
> [!NOTE]
> This step may take a while.

3. Ask Questions

```bash
python query.py
```
The assistant stays interactive and informs the user if no answer could be retrieved. The user can choose to see the retrieval chunks and their distance, or hide them.

Example:
```
Show retrieval similarity score and distance? (Y/N): n
Ask a question or type 'STOP' to exit: who wrote the micro paper?
---------------------------------------------------------------------------
Completed in 0.82s
Frank Elavsky, Carnegie Mellon University, fje@cmu.edu (Source: The_Micro-Paper.pdf, p. 1)
---------------------------------------------------------------------------
Ask a question or type 'STOP' to exit: stop
You've exited the program.
```

4. Run Evaluation

Update `evaluate.py` update the `QNA_SET` constant with questions and answers from your data/pdfs documents. Then, run:

```bash
python evaluate.py
```
The evaluation dashboard runs 15 test questions across multiple k values and reports retrieval accuracy and latency.

## Evaluation results
Retrieval accuracy was measured across three chunk sizes and five k values. 

**Key finding**: chunk size of 200 words with k=5 is the optimal configuration, hitting the 80% accuracy target with the smallest retrieval footprint.

Chunk Size	k=1	k=3	k=5	k=8	k=10	Failed
200 words	40%	73%	80%	87%	87%	    0%
400 words	27%	60%	60%	87%	87%	    6.7%
600 words	13%	40%	67%	73%	80%	    13.3%

**Observations**:

- Smaller chunks produce more focused embeddings that align more closely with short natural language queries. At k=3, accuracy drops from 73% to 60% to 40% as chunk size increases from 200 to 400 to 600 words.

- Larger chunks hurt low-k retrieval the most. At k=1, accuracy collapses from 40% to 27% to 13% as chunk size grows.

- Diminishing returns beyond k=8. Accuracy at k=10 equals k=8 across almost all configurations, suggesting the correct chunk is almost always within the top 8 results if it exists at all. Setting k above 8 adds prompt length and token cost without accuracy gain.


## Limitations

- Scanned PDFs are not supported. Only born-digital PDFs (created directly from software like LaTeX or Word) are supported. Scanned documents would require an additional OCR step.

- Named entity extraction (NER) is implemented but not yet used for filtering. spaCy NER runs on every chunk during ingestion and entities are stored in ChromaDB metadata, but entity-based query filtering is not yet implemented.

- Multi-hop questions perform poorly. Questions requiring reasoning across multiple documents simultaneously are a known limitation of standard RAG architectures.

- Chat history is not retained, every question has to be posed anew.

- Similarity scores are lower than expected. Cosine similarity between short natural language queries and long academic passages typically ranges between 0.28–0.58 in this system.

## What I Learned

This project was my first hands-on experience with NLP and AI engineering. Some of the key lessons I learned were:

- Chunking strategy matters a lot. The evaluation data showed that chunk size had a larger impact on retrieval accuracy than any other parameter. Getting chunking right is thereof very important.

- I deliberately avoided using LlamaIndex's automated query engine and built the retrieval and generation steps manually. This meant I could understand what is happening under the hood in better details and get more say in my system settings. It also allowed me to run my evaluation dashboard on multiple test cases to draw some conclusions. 

- Evaluation is an important part of AI engineering. Most tutorials show a working demo. But building my evaluation dashboard helped me determine if my system is actually improving, and why. 

- Small models are a real constraint. Running a 1B parameter model locally produced noticeably worse results than a 70B model via API. Understanding the tradeoff and choosing Groq-cloud-model quality over Ollama-local-privacy was an architectural decision I took for my NLP system.
