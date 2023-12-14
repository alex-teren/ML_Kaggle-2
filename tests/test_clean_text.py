# This test file contains unit tests for the `clean_text` function.

from src.clean_text import clean_text

def test_clean_text():
    assert clean_text("Example text") == "example text"