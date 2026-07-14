# Unit test of functions in query.py

import unittest 
from query import embed_query, similarity_search
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
        self.collection.delete(ids=self.collection.get()["ids"]) \
            if self.collection.count() > 0 else None

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

    def test_no_embedded_query(self):
        """Lack of embedded query should result in an empty output."""
        result = similarity_search(1, [], self.collection)
        self.assertIsNone(result)

    def test_invalid_k(self):
        """An invalid value of k should result in an empty output."""
        result = similarity_search(0, [0.22, 0.22, 0.22], self.collection)
        self.assertIsNone(result)

    def test_valid_inputs(self):
        """Result should contain distances, metadatas, and documents."""
        query = "Is attention mechanism used in transformer models?"
        embedded_query = embed_query(query, self.embedder)
        result = similarity_search(1, embedded_query, self.collection)
        
        self.assertIn("distances", result)
        self.assertIn("metadatas", result)
        self.assertIn("documents", result)


if __name__ == '__main__':
    unittest.main()
        