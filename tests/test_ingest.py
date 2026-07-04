# Unit test of functions in ingest.py

import unittest
import os
import tempfile

from ingest import extract_uploads, is_file_valid, clean_text, extract_pages, chunk_text

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



if __name__ == "__main__":
    unittest.main(verbosity=2)