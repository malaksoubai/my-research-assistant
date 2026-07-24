# Evaluation dashboard

import time
import json

from ingest import load_tools
from query import query_pipeline, retrieve_relevant_results, similarity_search

# TODO dict qna should include a question without an answer
QNA_SET = [
    # Cross-Age_LFW.pdf
    {
        "id": 1, 
        "question": "What does LFW stand for?",
        "expected_answer": "Labeled Faces in the Wild",
        "expected_filename": "Cross-Age_LFW.pdf",
        "expected_pages": [1, 9]
    },
    {
        "id": 2, 
        "question": "What paradigms are used in face recognition?",
        "expected_answer": "Identification and verification",
        "expected_filename": "Cross-Age_LFW.pdf",
        "expected_pages": [1, 2, 3]
    },
    {
        "id": 3, 
        "question": "Why was CALFW created?",
        "expected_answer": "To add age gap variation to face verification",
        "expected_filename": "Cross-Age_LFW.pdf",
        "expected_pages": [1, 2]
    },
    {
        "id": 4, 
        "question": "How many positive face pairs does CALFW contain?",
        "expected_answer": "3000",
        "expected_filename": "Cross-Age_LFW.pdf",
        "expected_pages": [1]
    },
    # Multi-Meta-RAG.pdf
    {
        "id": 5, 
        "question": "What does RAG stand for?",
        "expected_answer": "Retrieval-Augmented Generation",
        "expected_filename": "Multi-Meta-RAG.pdf",
        "expected_pages": [1]
    },
    {
        "id": 6, 
        "question": "What are the two main challenges of large language models?",
        "expected_answer": "static knowledge and generative hallucination",
        "expected_filename": "Multi-Meta-RAG.pdf",
        "expected_pages": [1]
    },
    {
        "id": 7, 
        "question": "What benchmark does Multi-Meta-RAG improve results on?",
        "expected_answer": "MultiHop-RAG",
        "expected_filename": "Multi-Meta-RAG.pdf",
        "expected_pages": [1, 2]
    },
    {
        "id": 8, 
        "question": "What two metadata fields does Multi-Meta-RAG extract from queries?",
        "expected_answer": "Article source and publication date",
        "expected_filename": "Multi-Meta-RAG.pdf",
        "expected_pages": [3]
    },
    # The_Emotional_Voices_Database.pdf
    {
        "id": 9, 
        "question": "How many emotion classes does the emotional voices database cover?",
        "expected_answer": "5",
        "expected_filename": "The_Emotional_Voices_Database.pdf",
        "expected_pages": [1]
    },
    {
        "id": 10, 
        "question": "What is the AmuS database used for?",
        "expected_answer": "Amused speech synthesis",
        "expected_filename": "The_Emotional_Voices_Database.pdf",
        "expected_pages": [3]
    },
    {
        "id": 11, 
        "question": "What emotions were chosen to cover a diverse space in the Russel Circumplex?",
        "expected_answer": "Amusement, anger, sleepiness, disgust and neutral",
        "expected_filename": "The_Emotional_Voices_Database.pdf",
        "expected_pages": [4]
    },
    {
        "id": 12, 
        "question": "What method was used to evaluate the MLP system?",
        "expected_answer": "CMOS perception test",
        "expected_filename": "The_Emotional_Voices_Database.pdf",
        "expected_pages": [1]
    },
    # The_Micro-Paper.pdf
    {
        "id": 13, 
        "question": "Who wrote the Macro paper?",
        "expected_answer": "Frank Elavsky",
        "expected_filename": "The_Micro-Paper.pdf",
        "expected_pages": [1]
    },
    {
        "id": 14,
        "question": "What is a micro-paper's goal?",
        "expected_answer": "Ideas for the sake of generative work, conversation, and inspiration.",
        "expected_filename": "The_Micro-Paper.pdf",
        "expected_pages": [1]
    },
    {
        "id": 15,
        "question": "why shouldn’t I write a blog or twitter thread for my paper?",
        "expected_answer": "Blogs and social media posts raise concerns about archival quality and trustworthiness. ",
        "expected_filename": "The_Micro-Paper.pdf",
        "expected_pages": [2]
    }
]

def get_valid_k(prompt: str) -> int:
    """Helper function for validating the k value."""
    while True:
        raw = input(prompt)
        try:
            k = int(raw)
        except ValueError:
            print("Please enter a whole number.")
            continue
        if 1 <= k <= 10:
            return k
        
        print("Value must be within [1, 10]. Please try again.")    

def run_one_k(k: int, embedder, collection) -> dict:
    """Runs all questions given one k value.
    Returns a result dict with scores and latency."""
    print("=" * 75) 
    print(f"    [STATUS:STARTED]  RUNNING FOR K={k}...")
    
    latency: float = 0.0
    pages_accuracy = 0
    failed = 0


    for q in QNA_SET:
        print("-" * 75) 
        start = time.time()
        print(q["id"])
        question = q["question"]
        pages = q["expected_pages"]

        results = similarity_search(k=k, input=question, embedder=embedder, collection=collection, show_stat=False)
        relevant_res = retrieve_relevant_results(results=results)

        latency += float(f"{time.time()-start:.2f}")

        if not relevant_res:
            # less accurate than run_one_k_with_llm
            # because it does not wait take in consideration LLM input, 
            # which could return None
            failed += 1 
            continue

        chunks = relevant_res["metadatas"]
        for chunk in chunks:
            page = chunk["page"]
            if page in pages:
                pages_accuracy += 1
                break   

    latency = latency / len(QNA_SET)
    accuracy = pages_accuracy / len(QNA_SET)
    failed = failed / len(QNA_SET) 

    return latency, accuracy, failed

# XXX untested + underdeveloped compared to run_one_k()
def run_one_k_with_llm(k: int, embedder, collection, llm) -> dict:
    """Runs all questions given one k value.
    Returns a result dict with scores and latency."""
    print(f"    [STATUS:STARTED]  RUNNING FOR K={k}...")
    latency: float = 0.0
    pages_accuracy = 0
    failed = 0


    for q in QNA_SET:
        print(q["id"])
        question = q["question"]
        pages = q["expected_pages"]

        response, time, relevant_res = query_pipeline(show_stat="n", query=question, k=k, embedder=embedder, collection=collection, llm=llm)
        latency += time
        if not relevant_res:
            failed += 1 
            continue

        chunks = relevant_res["metadatas"]
        for chunk in chunks:
            page = chunk["page"]
            if page in pages:
                pages_accuracy += 1
                break

        print("=" * 75)    

    latency = latency / len(QNA_SET)
    accuracy = pages_accuracy / len(QNA_SET)
    failed = failed / len(QNA_SET)         
    return latency, accuracy, failed


def print_summary(run_results: list[dict]) -> None:
    """Print results to the dashboard"""
    print ("-" * 70)
    print("EVALUATION RESULTS")
    print("-" * 70)
    print(f'{"k":<5} {"Avg accuracy":<15} {"Avg passed":<15} {"Avg failed":<15} {"Avg latency (sec)"}')
    print("-" * 70)
    for res in run_results:
        k = res['k']
        avg_acc = f"{res['accuracy'] * 100:.2f}%"
        avg_fail = f"{res['failed'] * 100:.2f}%"
        avg_pss = f"{(100 - (res['failed'] * 100)):.2f}%"
        avg_latency = f"{res['latency'] * 100:.2f}s"
        print(f"{(k):<5} {avg_acc:<15} {avg_pss:<15} {avg_fail:<15} {avg_latency}")


def evaluate_pipeline() -> dict:
    """Scores the system output against pre-registered QnA.""" 

    print("EVALUATION DASHBOARD\n")

    print("What top-k values would you like to compare? (k must be within [1, 10])")
    ks = []
    i = 1
    while len(ks) < 5:
        k = get_valid_k(f"k{i} = ")
        if k in ks:
            print(f"Value already added to list of ks {ks}. Please add a different one.")
            continue

        ks.append(k)
        i += 1

        if len(ks) < 5:
            more = input("Add another k value? (Y/N): ").strip().lower()
            if more.lower() == 'n':
                break
        

    print(f"The system will evaluate the following k values {ks}.")

    #load necessary tools
    nlp, embedder, collection, llm = load_tools()

    if collection.count() == 0:
        print("ERROR: No documents ingested. Run ingest.py first.")
        print(f"    [STATUS:FAILED]  VECTOR NOT EMBEDDED.")
        return

    # run questions and get sys output from query.py
    run_results = []
    for k in ks:
        try:
            latency, accuracy, failed = run_one_k(k = k, embedder=embedder, collection=collection)
            # latency, accuracy, failed = run_one_k_with_llm(k = k, embedder=embedder, collection=collection, llm=llm)
        except Exception as e:
            print(f'Error occured: {e}')
            return
        
        run_results.append({'k': k, 'latency': latency, 'accuracy': accuracy, 'failed': failed})
    
    # output logs in a dashboard
    print_summary(run_results)
    
    return


if __name__ == "__main__":
    evaluate_pipeline()
