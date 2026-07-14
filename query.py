# The goals of this script:
# 1. Query embedding
# 2. Similarity Search - top-k chunks + similarity score
# 3. Wire in LlamaIndex 

import config

from ingest import load_tools

def embed_query(input: str, embedder) -> list[float]:
    """Embed the query."""
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


def similarity_search(k: int, embedded_query: list[float], collection) -> None | list[dict]:
    """Performs top-k similarity search.
    Prints similarity score of top-k results."""

    if k < 1 or k > 5:
        print(f"    [STATUS:ERROR]  K is not an acceptable value of range [1, 5].")
        return None

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

    print("-" * 65)
    print("SIMILARITY SEARCH RESULTS")
    print("-" * 65)
    print(f'{"#":<5} {"Filename":<30} {"Page":<5} {"Distance":<10} {"Similarity"}')
    print("-" * 65)

    for i in range(len(distances)):
        d = round(distances[i], 4) # rounding only distance would yield to 0 or 1
        s = round(1 - d, 4)
        filename = metadatas[i]["filename"]
        page = metadatas[i]["page"]
        print(f"{(i + 1):<5} {filename:<30} {page:<5} {d:<10} {s}")
    
    print(f"Texts used: {documents}\n")
    
    return results

# --------------------------------------------------
# MAIN FUNCTION
# --------------------------------------------------

def main() -> None:
    """Runs the query pipeline to retrieve top-k similar chunks."""
    nlp, embedder, collection = load_tools()

    print("Your Research Assistant is read. Type 'STOP' to exit.\n")

    while True:
        query = input("Ask a question: ")

        if query.lower() in ('stop', 'quit', 'exit'):
            print("You've exited the program.")
            break

        embedded_query = embed_query(query)
        relevant_embeds = similarity_search(k=3, embedded_query=embedded_query, collection=collection)

        # TODO: wire in LlamaIndex and print result



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
#         embedded_query = embed_query(query, embedder=embedder)
#         output = similarity_search(2, embedded_query=embedded_query, collection=collection)

