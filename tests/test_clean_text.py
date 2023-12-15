# This test file contains unit tests for the `clean_text` function.

from src.clean_text import clean_text


def test_lowercase():
    assert clean_text("TeST") == "test"


def test_remove_html():
    assert clean_text(
        "<html>Content inside HTML tag</html>") == "content inside html tag"


def test_multiple_html_tags():
    assert clean_text(
        "<p>First paragraph</p> <p>Second paragraph</p>") == "first paragraph second paragraph"


def test_remove_url():
    assert clean_text("Follow link https://example.com") == "follow link"


def test_multiple_urls():
    assert clean_text(
        "Follow https://example.com and http://example.org") == "follow"


def test_empty_string():
    assert clean_text("") == ""


def test_remove_stopwords():
    assert clean_text("This is a test") == "test"


def test_lemmatization():
    # Assuming 'running' remains unchanged after lemmatization
    assert clean_text("running") == "running"


def test_non_english_characters():
    assert clean_text(
        "Café and Crème brûlée are delicious") == "café crème brûlée delicious"


def test_retain_essential_punctuation():
    assert clean_text("Hello, world.") == "hello, world."
