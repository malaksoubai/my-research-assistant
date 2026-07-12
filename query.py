# The goals of this script:
# 1. Query embedding
# 2. Similarity Search - top-k chunks + similarity score


from chromadb import Knn

def embed_query(input: str, embedder) -> list[float]:
    """Embed the query."""
    # no input, or invalid input
    if not input:
        raise ValueError("Input Error: No query was found.")
    
    try:
        embedded_query = embedder.encode(input)
    except Exception as e:
        print(f"ERROR occurred: {e}")
        print(f"    [STATUS:FAILED]  QUERY NOT EMBEDDED.")
        return

    print(f"    [STATUS:SUCCESS]  QUERY EMBEDDED.")

    return embedded_query


def similarity_search(k: int, query_text: str, embedded_query, collection) ->  list[dict]:
    """Performs top-k similarity search.
    Prints similarity score of top-k results."""

    if not query_text:
        print(f"    [STATUS:ERROR]  K is not an acceptable value of range [1, 5].")
        return

    if k < 1 or k > 5:
        print(f"    [STATUS:ERROR]  K is not an acceptable value of range [1, 5].")
        return
    
    results = collection.query(
        query_text=query_text,
        n_result = k,
        include = ["distances", "metadatas"],
    )
    
    # Knn(
    #     query = embedded_query, 
    #     key = "#embeddings", #K.EMBEDDING ?
    #     limit = k, 
    #     return_rank = False
    # )
