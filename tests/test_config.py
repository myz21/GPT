import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config.config import Config


class TestConfigModes:
    """All mode definitions in Config must be valid."""

    def test_tiny_mode_has_required_keys(self):
        cfg = Config.MODEL_CONFIGS["tiny"]
        for key in ("n_embd", "n_head", "n_layer", "batch_size", "max_iters", "desc"):
            assert key in cfg, f"tiny config missing key: {key}"

    def test_small_mode_values_are_positive(self):
        cfg = Config.MODEL_CONFIGS["small"]
        assert cfg["n_embd"] > 0
        assert cfg["n_head"] > 0
        assert cfg["n_layer"] > 0
        assert cfg["batch_size"] > 0
        assert cfg["max_iters"] > 0

    def test_medium_n_head_divides_n_embd(self):
        cfg = Config.MODEL_CONFIGS["medium"]
        assert cfg["n_embd"] % cfg["n_head"] == 0, \
            "n_embd must be divisible by n_head (head_size must be integer)"

    def test_large_n_head_divides_n_embd(self):
        cfg = Config.MODEL_CONFIGS["large"]
        assert cfg["n_embd"] % cfg["n_head"] == 0

    def test_all_modes_have_unique_params(self):
        sizes = set()
        for name, cfg in Config.MODEL_CONFIGS.items():
            n = cfg["n_embd"] * cfg["n_layer"]
            sizes.add(n)
        assert len(sizes) == len(Config.MODEL_CONFIGS), \
            "each mode must have a unique parameter count"


class TestConfigDevice:
    """Device auto-detection must work correctly."""

    def test_device_is_string(self):
        assert isinstance(Config.device, str)

    def test_device_is_cpu_or_cuda(self):
        assert Config.device in ("cpu", "cuda")


class TestConfigPaths:
    """Config paths must be valid."""

    def test_input_path_is_relative(self):
        assert Config.input_path.startswith("./data/")

    def test_model_save_path_has_placeholders(self):
        assert "{0}" in Config.model_save_path or "{}" in Config.model_save_path or "{0}_{1}" in Config.model_save_path or "{}_{}" in Config.model_save_path

    def test_bigram_save_path_ends_with_pth(self):
        assert Config.bigram_save_path.endswith(".pth")


class TestConfigHyperparameters:
    """Hyperparameters must be within reasonable ranges."""

    def test_learning_rate_is_reasonable(self):
        assert 1e-5 <= Config.learning_rate <= 1.0

    def test_dropout_is_between_zero_and_one(self):
        assert 0.0 <= Config.dropout <= 1.0

    def test_weight_decay_is_non_negative(self):
        assert Config.weight_decay >= 0.0

    def test_grad_clip_is_positive(self):
        assert Config.grad_clip > 0.0

    def test_min_lr_is_less_than_max_lr(self):
        assert Config.min_lr < Config.learning_rate

    def test_warmup_iters_less_than_max_iters(self):
        assert Config.warmup_iters < Config.max_iters

    def test_block_size_is_positive(self):
        assert Config.block_size > 0
