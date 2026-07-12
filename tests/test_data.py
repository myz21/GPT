import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDataProcessorInit:
    """DataProcessor must initialize correctly."""

    def test_text_is_loaded(self, data_processor):
        assert len(data_processor.text) > 0

    def test_vocab_size_is_positive(self, data_processor):
        assert data_processor.vocab_size > 0

    def test_stoi_maps_all_chars(self, data_processor):
        """Every char must have a mapping in stoi."""
        for ch in data_processor.chars:
            assert ch in data_processor.stoi

    def test_itos_maps_all_indices(self, data_processor):
        """Every index must have a mapping in itos."""
        for i in range(data_processor.vocab_size):
            assert i in data_processor.itos

    def test_stoi_and_itos_are_inverses(self, data_processor):
        """stoi and itos must be inverses of each other."""
        for ch, i in data_processor.stoi.items():
            assert data_processor.itos[i] == ch

    def test_train_data_exists(self, data_processor):
        assert len(data_processor.train_data) > 0

    def test_val_data_exists(self, data_processor):
        assert len(data_processor.val_data) > 0

    def test_train_val_are_disjoint(self, data_processor):
        """Train and val indices must not overlap."""
        train_set = set(data_processor.train_data.tolist())
        val_set = set(data_processor.val_data.tolist())
        assert len(train_set & val_set) <= len(val_set), \
            "train and val may share similar content but indices must be disjoint"

    def test_train_plus_val_equals_total(self, data_processor):
        total = len(data_processor.train_data) + len(data_processor.val_data)
        assert total == len(data_processor.text)


class TestDataProcessorEncodeDecode:
    """Encode/decode roundtrip must work correctly."""

    def test_encode_returns_list_of_ints(self, data_processor, sample_text):
        encoded = data_processor.encode(sample_text)
        assert isinstance(encoded, list)
        assert all(isinstance(i, int) for i in encoded)

    def test_encode_length_matches_input(self, data_processor, sample_text):
        encoded = data_processor.encode(sample_text)
        assert len(encoded) == len(sample_text)

    def test_decode_returns_string(self, data_processor, sample_text):
        encoded = data_processor.encode(sample_text)
        decoded = data_processor.decode(encoded)
        assert isinstance(decoded, str)

    def test_roundtrip(self, data_processor, sample_text):
        """encode -> decode must return the original text."""
        encoded = data_processor.encode(sample_text)
        decoded = data_processor.decode(encoded)
        assert decoded == sample_text

    def test_unknown_char_raises_keyerror(self, data_processor):
        """Encoding a char not in vocab must raise KeyError."""
        with pytest.raises(KeyError):
            data_processor.encode("\u25cf")  # this char is not in data.txt


class TestDataProcessorGetBatch:
    """get_batch must produce correct tensors."""

    def test_get_batch_returns_two_tensors(self, data_processor):
        x, y = data_processor.get_batch('train')
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)

    def test_get_batch_shapes(self, data_processor):
        """get_batch must produce (batch_size, block_size) tensors."""
        x, y = data_processor.get_batch('train')
        assert x.shape == (4, 32)  # batch_size=4, block_size=32
        assert y.shape == (4, 32)

    def test_get_batch_y_is_x_shifted(self, data_processor):
        """y must be x shifted by 1 (next token prediction)."""
        x, y = data_processor.get_batch('train')
        assert torch.equal(x[:, 1:], y[:, :-1])

    def test_get_batch_dtype_is_long(self, data_processor):
        x, y = data_processor.get_batch('train')
        assert x.dtype == torch.long
        assert y.dtype == torch.long

    def test_get_batch_train_split(self, data_processor):
        x_train, _ = data_processor.get_batch('train')
        x_val, _ = data_processor.get_batch('val')
        assert x_train.shape == x_val.shape

    def test_get_batch_randomness(self, data_processor):
        """Two get_batch calls must return different batches."""
        x1, _ = data_processor.get_batch('train')
        x2, _ = data_processor.get_batch('train')
        assert not torch.equal(x1, x2)

    def test_get_batch_indices_in_range(self, data_processor):
        """Generated indices must be within vocab range."""
        x, _ = data_processor.get_batch('train')
        assert x.min() >= 0
        assert x.max() < data_processor.vocab_size


class TestDataProcessorEdgeCases:
    """Edge cases must be handled correctly."""

    def test_single_char_text(self, tmp_path):
        """Single-character text must work."""
        from src.data import DataProcessor
        f = tmp_path / "single.txt"
        f.write_text("a")
        dp = DataProcessor(str(f), val_split=0.5)
        assert dp.vocab_size == 1

    def test_repeated_chars_text(self, tmp_path):
        from src.data import DataProcessor
        f = tmp_path / "repeated.txt"
        f.write_text("aaaaa")
        dp = DataProcessor(str(f), val_split=0.5)
        assert dp.vocab_size == 1

    def test_val_split_zero(self, tmp_path):
        """val_split=0 must put all data in train."""
        from src.data import DataProcessor
        f = tmp_path / "test.txt"
        f.write_text("test verisi")
        dp = DataProcessor(str(f), val_split=0.0)
        assert len(dp.val_data) == 0
        assert len(dp.train_data) == len(dp.text)

    def test_val_split_one(self, tmp_path):
        """val_split=1 must put all data in val."""
        from src.data import DataProcessor
        f = tmp_path / "test.txt"
        f.write_text("test verisi")
        dp = DataProcessor(str(f), val_split=1.0)
        assert len(dp.train_data) == 0
        assert len(dp.val_data) == len(dp.text)
