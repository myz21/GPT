import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import Config
from src.data import DataProcessor


def patch_config(monkeypatch, overrides):
    for key, value in overrides.items():
        monkeypatch.setattr(Config, key, value)


@pytest.fixture
def test_config(monkeypatch):
    patch_config(monkeypatch, {
        "mode": "tiny",
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 3,
        "batch_size": 4,
        "max_iters": 10,
        "block_size": 32,
        "eval_iters": 2,
        "eval_interval": 5,
        "dropout": 0.0,
        "warmup_iters": 2,
        "lr_decay_iters": 10,
        "device": "cpu",
    })
    return Config


@pytest.fixture
def data_processor(monkeypatch):
    patch_config(monkeypatch, {
        "mode": "tiny",
        "n_embd": 64,
        "n_head": 4,
        "n_layer": 3,
        "batch_size": 4,
        "block_size": 32,
        "device": "cpu",
    })
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data.txt")
    return DataProcessor(data_path, val_split=0.1)


@pytest.fixture
def sample_text():
    return "Merhaba dünya"


@pytest.fixture
def seed():
    torch.manual_seed(42)
