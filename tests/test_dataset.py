import os

import pytest

from config.config import Config

FULL_EXISTS = os.path.exists(Config.input_path)
SMALL_EXISTS = os.path.exists(Config.havadis_small_path)
ANY_EXISTS = FULL_EXISTS or SMALL_EXISTS

pytestmark = pytest.mark.skipif(
    not ANY_EXISTS,
    reason="No dataset found. Run: python scripts/download_and_prepare.py (--small)",
)


def _get_dataset_path():
    if FULL_EXISTS:
        return Config.input_path
    return Config.havadis_small_path


@pytest.fixture
def dataset_path():
    return _get_dataset_path()


@pytest.fixture
def dataset_text(dataset_path):
    with open(dataset_path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def articles(dataset_text):
    return dataset_text.split("HABER SONU\n\n")


class TestDatasetExists:
    def test_file_exists(self, dataset_path):
        assert os.path.exists(dataset_path)

    def test_file_not_empty(self, dataset_text):
        assert len(dataset_text) > 0

    def test_file_size_minimum(self, dataset_path):
        size_mb = os.path.getsize(dataset_path) / 1e6
        assert size_mb >= 1, f"Dataset too small: {size_mb:.1f} MB (expected >= 1 MB)"


class TestDatasetFormat:
    def test_articles_have_title(self, articles):
        for i, article in enumerate(articles):
            if not article.strip():
                continue
            assert article.strip().startswith("Başlık:"), (
                f"Article {i} missing 'Başlık:' prefix"
            )

    def test_articles_have_content(self, articles):
        for i, article in enumerate(articles):
            if not article.strip():
                continue
            assert "İçerik:" in article, (
                f"Article {i} missing 'İçerik:' field"
            )

    def test_articles_end_with_marker(self, articles):
        for i, article in enumerate(articles):
            if not article.strip():
                continue
            assert article.strip().endswith("HABER SONU"), (
                f"Article {i} missing 'HABER SONU' marker"
            )

    def test_no_empty_articles(self, articles):
        non_empty = [a for a in articles if a.strip()]
        assert len(non_empty) > 0, "All articles are empty"

    def test_minimum_article_count(self, articles):
        count = len([a for a in articles if a.strip()])
        assert count >= 100, f"Too few articles: {count} (expected >= 100)"


class TestDatasetEncoding:
    def test_utf8_encoding(self, dataset_text):
        assert isinstance(dataset_text, str)

    def test_turkish_characters_present(self, dataset_text):
        turkish_chars = set("çğıöşüÇĞİÖŞÜ")
        found = turkish_chars.intersection(set(dataset_text))
        assert len(found) > 0, "No Turkish characters found — wrong dataset?"

    def test_no_garbled_text(self, dataset_text):
        replacement_chars = dataset_text.count("�")
        assert replacement_chars == 0, f"Found {replacement_chars} garbled characters (encoding error)"


class TestDatasetContent:
    def test_article_minimum_length(self, articles):
        short = []
        for i, article in enumerate(articles):
            if not article.strip():
                continue
            if len(article.strip()) < 100:
                short.append(i)
        assert len(short) == 0, f"Articles too short: {short}"

    def test_no_duplicate_consecutive_articles(self, articles):
        seen = set()
        duplicates = []
        for i, article in enumerate(articles):
            key = article.strip()[:100]
            if key in seen:
                duplicates.append(i)
            seen.add(key)
        assert len(duplicates) == 0, f"Duplicate articles found: {duplicates}"

    def test_vocab_coverage(self, dataset_text):
        chars = set(dataset_text)
        assert len(chars) >= 30, f"Too few unique characters: {len(chars)} (expected >= 30)"
