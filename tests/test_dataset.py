import os

import pytest

from config.config import Config

DATASET_EXISTS = os.path.exists(Config.input_path)

pytestmark = pytest.mark.skipif(
    not DATASET_EXISTS,
    reason="Havadis dataset not downloaded. Run: python scripts/download_and_prepare.py",
)


@pytest.fixture
def dataset_text():
    with open(Config.input_path, encoding="utf-8") as f:
        return f.read()


@pytest.fixture
def articles(dataset_text):
    return dataset_text.split("HABER SONU\n\n")


class TestDatasetExists:
    def test_file_exists(self):
        assert os.path.exists(Config.input_path)

    def test_file_not_empty(self, dataset_text):
        assert len(dataset_text) > 0

    def test_file_size_minimum(self):
        size_mb = os.path.getsize(Config.input_path) / 1e6
        assert size_mb >= 10, f"Dataset too small: {size_mb:.1f} MB (expected >= 10 MB)"


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
        assert count >= 1000, f"Too few articles: {count} (expected >= 1000)"


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
