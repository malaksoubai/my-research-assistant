# Unit test of functions in query.py

import unittest 
from sentence_transformers import SentenceTransformer
import chromadb

from query import embed_query, similarity_search
from test_ingest import ModelTestCase

class TestEmbedQuery(ModelTestCase):

    def test_embed_no_query(self):
        """Lack of input should raise value error."""
        with self.assertRaises(ValueError):
            embed_query("", self.embedder)

    def test_embed_query_returns_list(self):
        """Output must of  type list."""
        input = "The cat is agitated."
        result = embed_query(input, self.embedder)
        self.assertIsInstance(result, list)

    def test_embed_query(self):
        """Output must be a populated list."""
        input = "The cat is agitated."
        result = embed_query(input, self.embedder)
        self.assertGreater(len(result), 0)

class TestSimilaritySearch(ModelTestCase):
    """Test if similarity_search() function handles edge cases."""

    def test_no_embedded_query(self):
        """Lack of embedded query should result in an empty output."""
        result = similarity_search(1, [], self.collection)
        self.assertEqual(result, [])

    def test_invalid_k(self):
        """An invalid value of k should result in an empty output."""
        result = similarity_search(0, [0.22, 0.22, 0.22], self.collection)
        self.assertEqual(result, [])


if __name__ == '__main__':
    unittest.main()
        