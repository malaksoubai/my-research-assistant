# Unit test of functions in ingest.py

import shutil
import unittest
import os
import tempfile
from sentence_transformers import SentenceTransformer
import spacy
import chromadb

import config
from ingest import embed_and_store, extract_entities, extract_uploads, is_file_valid, clean_text, chunk_text

# --------------------------------------------------
# 1. Read all PDFs from a folder and validate them
# --------------------------------------------------

class TestExtractUploads(unittest.TestCase):

    def setUp(self):
        """Creates temp dir for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Removes temp dir after testing."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_empty_dir(self):
        """Empty folder should return an empty list."""
        result = extract_uploads(self.temp_dir)
        self.assertEqual(result, [])
    
    def test_non_empty_dir(self):
        """Populated dir should return a list of files in dir."""
        temp_file = os.path.join(self.temp_dir, "file.pdf") # creates file path
        open(temp_file, "w").close()    # creates empty (write) file
        result = extract_uploads(self.temp_dir)
        self.assertEqual(len(result), 1)


class TestIsFileValid(unittest.TestCase):

    def setUp(self):
        """Creates temp dir for testing."""
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """Removes temp dir after testing."""
        import shutil
        shutil.rmtree(self.temp_dir)

    def test_non_pdf_file(self):
        """Returns False if file not a PDF."""
        temp_file = os.path.join(self.temp_dir, "file.txt")
        open(temp_file, "w").close()
        result = is_file_valid(temp_file)
        self.assertFalse(result[0])

    def test_empty_file(self):
        """Returns False if file is empty."""
        temp_file = os.path.join(self.temp_dir, "file.pdf")
        open(temp_file, "w").close()
        result, _ = is_file_valid(temp_file)
        self.assertFalse(result)

    def test_outputs_tuple(self):
        """Validate return type of output."""
        temp_file = os.path.join(self.temp_dir, "paper.txt")
        open(temp_file, "w").close()
        result = is_file_valid(temp_file)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)
        self.assertIsInstance(result[0], bool)
        self.assertIsInstance(result[1], str)


# --------------------------------------------------
# 2. Extract text per page and clean it
# --------------------------------------------------

class TestCleanText(unittest.TestCase):

    def test_fi_ligature_chars(self):
        """ﬁ should become fi."""
        result = clean_text(text = "An ofﬁcer.")
        self.assertEqual(result, "An officer.")

    def test_fl_ligature_chars(self):
        """ﬂ should become fl."""
        result = clean_text(text = "A ﬂask.")
        self.assertEqual(result, "A flask.")

    def test_ff_ligature_chars(self):
        """ﬀ should become ff."""
        result = clean_text(text = "Huﬀing and puﬀing.")
        self.assertEqual(result, "Huffing and puffing.")
    
    def test_ffi_ligature_chars(self):
        """ﬃ should become ffi."""
        result = clean_text(text = "A coﬃn.")
        self.assertEqual(result, "A coffin.")
    
    def test_ffl_ligature_chars(self):
        """ﬄ should become ffl."""
        result = clean_text(text = "Baﬄed.")
        self.assertEqual(result, "Baffled.")

    def test_arxiv(self):
        """arXiv should be removed."""
        result = clean_text(text = "arXiv:1708.08197v1  [cs.CV]  28 Aug 2017")
        self.assertNotIn("arXiv", result)

    def test_multiple_spaces(self):
        """Collapse multiple spaces."""
        result = clean_text("Clean     text   ")
        self.assertEqual(result, "Clean text")

    def test_multiple_breaks(self):
        """Collapse multiple breaks."""
        result = clean_text("This\nclean\ntext.")
        self.assertEqual(result, "This clean text.")

    def test_empty_str(self):
        """Empty string should stay empty."""
        self.assertEqual(clean_text(""), "")


# --------------------------------------------------
# 3. Chunk text and attach metadata
# --------------------------------------------------

class TestChunkTest(unittest.TestCase):

    def setUp(self):
        """Create a fake page with words."""

        words = "word " * 900

        self.pages = [{"page_num": 1, "text": words},
                      {"page_num": 2, "text": "New page with little text."}]
        
        self.file_path = "tests/test_paper.pdf"

    def test_returns_list(self):
        """Output must be of type list."""
        result = chunk_text(self.pages, self.file_path)
        self.assertIsInstance(result, list)

    def test_metadata(self):
        """Output should include all metadata."""
        chunks = chunk_text(self.pages, self.file_path)
        metadata = {'text', 'filename', 'page_num', 'chunk_id'}
        for chunk in chunks:
            self.assertEqual(set(chunk.keys()), metadata, msg=f"Chunk missing keys: {chunk}")
        
    def test_correct_page_num(self):
        """Metadata should point at the correct page number."""
        chunks = chunk_text(self.pages, self.file_path) 
        # all chunks from page 1 should have page_num = 1
        page1_chunks = [c for c in chunks if c["page_num"] == 1]
        for chunk in page1_chunks:
            self.assertEqual(chunk["page_num"], 1)

        # all chunks from page 2 should have page_num = 2
        page2_chunks = [c for c in chunks if c["page_num"] == 2]
        for chunk in page2_chunks:
            self.assertEqual(chunk["page_num"], 2)

    def test_empty_page(self):
        """Empty pages should not produce any chunk."""
        empty_page = [{"page_num": 1, "text": ""}]

        result = chunk_text(empty_page, self.file_path)
        self.assertEqual(result, [])


# ──────────────────────────────────────────────
# Claude Code:
# Shared fixture: load models once for all model-dependent tests
# Runs once per test session, not once per test — keeps things fast
# ──────────────────────────────────────────────

class ModelTestCase(unittest.TestCase):
    """Base class that loads models once and shares them across subclasses.
    Any test class that needs real models should inherit from this."""

    @classmethod
    def setUpClass(cls):
        """Loads spaCy, Sentence Transformers, and a temporary ChromaDB
        collection once before any test in this class runs."""
        print("\nLoading models for integration tests (one-time cost)...")
        cls.nlp = spacy.load("en_core_web_sm")
        cls.embedder = SentenceTransformer("all-MiniLM-L6-v2")

        cls.client = chromadb.EphemeralClient()
        cls.collection = cls.client.get_or_create_collection("test_collection")

# --------------------------------------------------
# 4. spaCy named entity extraction per chunk
# --------------------------------------------------

class TestExtractEntities(ModelTestCase):

    def test_empty_input(self):
        """Empty strings should yield an empty string."""
        result = extract_entities(text=" ", nlp=self.nlp)
        self.assertEqual(result, "")

    def test_extract_organization_entity(self):
        """spaCy should identify known organization."""
        result = extract_entities("Google is an organization.", self.nlp)
        self.assertIn("Google", result)

    def test_extract_date_entity(self):
        """spaCy should identify known date."""
        result = extract_entities("I started this project on June 2026.", self.nlp)
        self.assertIn("June", result)

    def test_extract_person_entity(self):
        """spaCy should identify known person."""
        result = extract_entities("Mohammed IV is the King of Morocco.", self.nlp)
        self.assertIn("Mohammed", result)

    def test_extract_place_entity(self):
        """spaCy should identify known place."""
        result = extract_entities("London is the capital city of England.", self.nlp)
        self.assertIn("London", result)

# --------------------------------------------------
# 5. vectorize chunks and store in chromaDB
# --------------------------------------------------

class TestEmbedAndStore(ModelTestCase):

    def make_chunk(self, num: int) -> list[dict]:
        """Helper function: Creates num number of chunks in a list."""
        result = []
        for n in range(num):
            result.append({
                "text": "This is a test chunk containing data.",
                "filename": "Test-Chunk",
                "page_num": 1,
                "chunk_id": f"Test-Chunk_{n}"
            })
        return result

    def test_empty_chunk(self):
        """Empty chunk should yield to nothing stored in the db."""
        embed_and_store([], self.nlp, self.embedder, self.collection)
        self.assertEquals(self.collection.count(), 0)

    def test_two_chunks(self):
        """Two chunk should yield to two stored in the db."""
        chunks = self.make_chunk(2)
        embed_and_store(chunks, self.nlp, self.embedder, self.collection)
        self.assertEquals(self.collection.count(), 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)