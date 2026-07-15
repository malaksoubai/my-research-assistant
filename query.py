# The goals of this script:
# 1. Query embedding
# 2. Similarity Search - top-k chunks + similarity score
# 3. Wire in LlamaIndex 

import config
from ingest import load_tools

import time
from llama_index.llms.ollama import Ollama


# --------------------------------------------------
# 1. Query embedding
# --------------------------------------------------

def embed_query(input: str, embedder) -> list[float] | None:
    """Helper function to embed a query using a unimodal embedder."""
    # no input, or invalid input
    if len(input) == 0:
        raise ValueError("Input Error: No query was found.")
    
    try:
        embedded_query = embedder.encode(input)
    except Exception as e:
        print(f"ERROR occurred: {e}")
        print(f"    [STATUS:FAILED]  QUERY NOT EMBEDDED.")
        return

    print(f"    [STATUS:SUCCESS]  QUERY EMBEDDED.")

    return embedded_query.tolist()

# --------------------------------------------------
# 2. Similarity Search - top-k chunks + similarity score
# --------------------------------------------------

def similarity_search(k: int, input: str, embedder, collection) -> None | dict[str, list]:
    """Performs top-k similarity search.
    Prints similarity score of top-k results."""

    if k < 1 or k > 5:
        print(f"    [STATUS:ERROR]  K is not an acceptable value of range [1, 5].")
        # FIXME: instead of printing None, should I raise an error?
        return None
    
    embedded_query = embed_query(input, embedder)

    if embedded_query is None or len(embedded_query) == 0:
        print(f"    [STATUS:ERROR]  K is not an acceptable value of range [1, 5].")
        return None
    
    # results is a dict with "" keys and [[]] values
    results = collection.query(
        query_embeddings=[embedded_query],
        n_results = k,
        include = ["distances", "metadatas", "documents"],
    )

    distances = results["distances"][0]
    metadatas = results["metadatas"][0]
    documents = results["documents"][0]

    print(f"    [STATUS:SUCCESS]  CHUNKS RETRIEVED.")

    print("-" * 75)
    print("SIMILARITY SEARCH RESULTS")
    print("-" * 75)
    print(f'{"#":<5} {"Filename":<40} {"Page":<5} {"Distance":<10} {"Similarity"}')
    print("-" * 75)

    for i in range(len(distances)):
        d = round(distances[i], 4) # rounding only distance would yield to 0 or 1
        s = round(1 - d, 4)
        filename = metadatas[i]["filename"]
        page = metadatas[i]["page"]
        print(f"{(i + 1):<5} {filename:<40} {page:<5} {d:<10} {s}")
        
    
    print(f"Texts used: {documents}\n")
    
    return results

def retrieve_relevant_results(results: dict[str, list], threshold: float = 0.4) -> None | dict[str, list]:
    """Helper function used to validate output results by checking relevancy."""
    if results is None:
        return None
    
    filtered_documents = []
    filtered_metadatas = []

    for d, doc, meta in zip(
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0]
    ):
        similarity = 1 - d

        if similarity >= threshold:
            filtered_documents.append(doc)
            filtered_metadatas.append(meta)

    if len(filtered_documents) == 0:
        return {}
        
    return {"documents": filtered_documents, "metadatas": filtered_metadatas}

# --------------------------------------------------
# 3. Wire in LlamaIndex 
# --------------------------------------------------

SYSTEM_PROMPT = """
You are a research assistant. Answer the question using only the sources below. 

Your rules:
- Your answers are exclusively sourced from the data shared below.
- After every claim, support it using the following citation: (Source: <filename>, p. <page>)
- If the data shared with you does not answer the query, ONLY output: 'I could not find this in the uploaded documents.'
- Return Raw output with no additional comment or assumption.
"""

def build_prompt(query: str, relevant_results: None | dict[str, list], sys_prompt: str = SYSTEM_PROMPT) -> str:
    """Helper function that customizes the prompt for ollama."""
    sources = []
    for i, (doc, meta) in enumerate(zip(relevant_results["documents"], relevant_results["metadatas"]), start = 1):
        source = (
            f"--- source {i} ---\n"
            f"--- filename: {meta["filename"]} ---\n"
            f"--- page: {meta["page"]} ---\n"
            f"--- content: {doc} ---\n"
        )
        
        sources.append(source)
    
    sources_txt = "\n".join(sources)

    prompt = f"""{SYSTEM_PROMPT} 
            SOURCES: {sources_txt}
            QUERY: {query}
            ANSWER:"""

    return prompt

def generate_answer(query: str, relevant_results: dict[str, list]) -> str:
    """Wire LlamaIndex to generate answer from Ollama."""
    print("=" * 75)
    prompt = build_prompt(query, relevant_results)
    llm = Ollama(model=config.LLM_MODEL, request_timeout=120.0)

    print(f"    [STATUS:STARTED]  SYSTEM LOADING AN ANSWER.")
    print('Awaiting answer from ollama. Please wait, this may take some time...')
    start = time.time()

    # Sets the timeout to 120 seconds instead of the default 30 seconds
    response = llm.complete(prompt)
    print(f"Completed in {time.time()-start:.2f}s")

    return response
        

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main() -> None:
    """Runs the query pipeline to retrieve top-k similar chunks."""
    nlp, embedder, collection = load_tools()

    print("Your Research Assistant is read.\n")

    while True:
        print("-" * 75)
        query = input("Ask a question or type 'STOP' to exit: ")

        if query.lower() in ('stop', 'quit', 'exit'):
            print("You've exited the program.")
            break

        elif query:
            top_k = similarity_search(k=3, input=query, embedder=embedder, collection=collection)
            relevant_results = retrieve_relevant_results(results=top_k)

            if relevant_results == {}:
                print('\nI could not find this in the uploaded documents.\n')

            else:
                response = generate_answer(query, relevant_results)
                print(str(response))


if __name__ == "__main__":
    main()

# --------------------------------------------------
# SMOKE TESTS
# --------------------------------------------------
# Add smoke test for sanity checks here

# if __name__ == "__main__":
#     print("started SMOKE TESTING query.py file")
#     nlp, embedder, collection = load_tools()
#     queries = [
#         "What ingredients are needed for a chocolate cake recipe?"
#     ]

#     for query in queries:
#         print(f"\n\nQuery: {query}\n")
#         output = similarity_search(k=1, input=query, embedder=embedder, collection=collection)


