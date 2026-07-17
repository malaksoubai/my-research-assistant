# Unit test of functions in query.py

import unittest 
from query import embed_query, similarity_search, retrieve_relevant_results
from tests.test_ingest import ModelTestCase

class TestEmbedQuery(ModelTestCase):

    def test_embed_no_query(self):
        """Lack of input should raise value error."""
        with self.assertRaises(ValueError):
            embed_query("", self.embedder)

    def test_embed_query(self):
        """Output must of  type list."""
        input = "The cat is agitated."
        result = embed_query(input, self.embedder)

        self.assertIsInstance(result, list)
        self.assertGreater(len(result), 0)


class TestSimilaritySearch(ModelTestCase):
    # Credit: Claude code came up with setUp
    def setUp(self):
        """Populate collection with a few chunks before each test."""
        # clear first to avoid duplicate ID errors across tests
        self.collection.delete(ids=self.collection.get()["ids"]) if self.collection.count() > 0 else None

        fake_chunks = [
            {
                "text": "The transformer model uses attention mechanisms for NLP tasks.",
                "filename": "test_paper.pdf",
                "page_num": 1,
                "chunk_id": "test_paper.pdf_0"
            },
            {
                "text": "Face recognition is used to identify individuals from images.",
                "filename": "test_paper.pdf",
                "page_num": 2,
                "chunk_id": "test_paper.pdf_1"
            },
            {
                "text": "Emotional speech synthesis requires labeled audio datasets.",
                "filename": "test_paper.pdf",
                "page_num": 3,
                "chunk_id": "test_paper.pdf_2"
            }
        ]

        from ingest import embed_and_store
        embed_and_store(fake_chunks, self.nlp, self.embedder, self.collection)

    def test_invalid_k(self):
        """An invalid value of k should result in an empty output."""
        result = similarity_search(0, "Hello, world!", self.embedder, self.collection)
        self.assertIsNone(result)

    def test_valid_inputs(self):
        """Result should contain distances, metadatas, and documents."""
        query = "Is attention mechanism used in transformer models?"
        result = similarity_search(1, query, self.embedder, self.collection)
        
        self.assertIn("distances", result)
        self.assertIn("metadatas", result)
        self.assertIn("documents", result)

class TestRetrieveRelevantResults(unittest.TestCase):
    
    def test_no_results_arg(self):
        """Lack of results argument should yield to no output."""
        output = retrieve_relevant_results(None)
        self.assertIsNone(output)
    
    def test_revant_result(self):
        """Results should be over default threshold of 0.3 and contain two keys."""
        results = {
            "distances": [[0.1, 0.8, 0.7]],
            "documents": [["valid_doc_1", "non_valid_doc", "valid_doc_2"]],
            "metadatas": [[1, 2, 3]]
        }
        result = retrieve_relevant_results(results)

        self.assertNotIn("distances", result)
        self.assertIn("metadatas", result)
        self.assertIn("documents", result)

        self.assertNotIn("non_valid_doc", result["documents"])
        self.assertIn("valid_doc_1", result["documents"])
        self.assertIn("valid_doc_2", result["documents"])


if __name__ == '__main__':
    unittest.main()
        