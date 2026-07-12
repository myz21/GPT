import math
import os

import pytest
import torch

torch.set_num_threads(1)

from config.config import Config
from src.bigram import BigramModel
from src.data import DataProcessor
from src.model import GPTLanguageModel


TINY_PATCH = {
    "mode": "tiny",
    "n_embd": 64,
    "n_head": 4,
    "n_layer": 3,
    "batch_size": 4,
    "max_iters": 6,
    "block_size": 32,
    "eval_iters": 2,
    "eval_interval": 3,
    "dropout": 0.0,
    "warmup_iters": 2,
    "lr_decay_iters": 6,
    "min_lr": 1e-5,
    "device": "cpu",
    "seed": 42,
}


def _patch_config(monkeypatch):
    for key, value in TINY_PATCH.items():
        monkeypatch.setattr(Config, key, value)


def _make_dp():
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.txt")
    return DataProcessor(data_path, val_split=0.1)


def _train_gpt(dp, model, steps=6):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay
    )
    for _ in range(steps):
        xb, yb = dp.get_batch("train")
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if Config.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), Config.grad_clip)
        optimizer.step()


class TestPipelineDataToModel:
    def test_gpt_forward(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        model = GPTLanguageModel(dp.vocab_size).to(Config.device)

        xb, yb = dp.get_batch("train")
        logits, loss = model(xb, yb)

        assert logits.shape == (Config.batch_size * Config.block_size, dp.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)

    def test_bigram_forward(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        bigram = BigramModel(dp.vocab_size).to(Config.device)

        xb, yb = dp.get_batch("val")
        logits, loss = bigram(xb, yb)

        assert logits.shape == (Config.batch_size * Config.block_size, dp.vocab_size)
        assert loss is not None
        assert not torch.isnan(loss)


class TestPipelineTraining:
    def test_bigram_learning(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        bigram = BigramModel(dp.vocab_size).to(Config.device)
        optimizer = torch.optim.AdamW(bigram.parameters(), lr=1e-3)

        first_loss = None
        for i in range(100):
            xb, yb = dp.get_batch("train")
            _, loss = bigram(xb, yb)
            if i == 0:
                first_loss = loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        assert first_loss is not None
        assert not math.isinf(loss.item())
        assert not math.isnan(loss.item())

    def test_gpt_training_loop(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        model = GPTLanguageModel(dp.vocab_size).to(Config.device)

        _train_gpt(dp, model)

        model.eval()
        with torch.no_grad():
            xb, yb = dp.get_batch("val")
            _, loss = model(xb, yb)

        assert not torch.isnan(loss)
        assert loss.item() < math.log(dp.vocab_size)


class TestPipelineGeneration:
    def test_bigram_generate(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        bigram = BigramModel(dp.vocab_size).to(Config.device)

        bigram.eval()
        with torch.no_grad():
            ctx = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
            out = bigram.generate(ctx, max_new_tokens=50)
            text = dp.decode(out[0].tolist())

        assert isinstance(text, str)
        assert len(text) > 0

    def test_gpt_generate(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        model = GPTLanguageModel(dp.vocab_size).to(Config.device)

        _train_gpt(dp, model)

        model.eval()
        with torch.no_grad():
            ctx = torch.zeros((1, 1), dtype=torch.long, device=Config.device)
            out = model.generate(ctx, max_new_tokens=50)
            text = dp.decode(out[0].tolist())

        assert isinstance(text, str)
        assert len(text) > 0


class TestPipelineValidation:
    def test_gpt_beats_random_baseline(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        random_loss = math.log(dp.vocab_size)

        model = GPTLanguageModel(dp.vocab_size).to(Config.device)
        _train_gpt(dp, model)

        model.eval()
        with torch.no_grad():
            losses = torch.zeros(Config.eval_iters)
            for k in range(Config.eval_iters):
                xb, yb = dp.get_batch("val")
                _, loss = model(xb, yb)
                losses[k] = loss.item()
            gpt_loss = losses.mean().item()

        assert gpt_loss < random_loss

    def test_encode_decode_roundtrip(self, monkeypatch):
        _patch_config(monkeypatch)
        dp = _make_dp()
        original = "hello"
        assert dp.decode(dp.encode(original)) == original

    def test_checkpoint_save_load(self, monkeypatch, tmp_path):
        _patch_config(monkeypatch)
        dp = _make_dp()
        model = GPTLanguageModel(dp.vocab_size).to(Config.device)

        xb, yb = dp.get_batch("train")
        _, loss = model(xb, yb)
        loss.backward()

        save_path = str(tmp_path / "model.pth")
        torch.save({"model_state_dict": model.state_dict(), "loss": loss.item()}, save_path)

        ckpt = torch.load(save_path, weights_only=False)
        model2 = GPTLanguageModel(dp.vocab_size).to(Config.device)
        model2.load_state_dict(ckpt["model_state_dict"])

        model.eval()
        model2.eval()
        with torch.no_grad():
            xb2, yb2 = dp.get_batch("val")
            _, l1 = model(xb2, yb2)
            _, l2 = model2(xb2, yb2)

        assert abs(l1.item() - l2.item()) < 1e-5
