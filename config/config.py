import torch


class Config:
    seed = 1337
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # === Dataset ===
    dataset = "havadis"  # havaids, data.txt, custom
    havaids_raw_path = "./data/havadis_raw.txt"
    input_path = "./data/havadis_processed.txt"

    # === Havadis Processing ===
    havaids_max_articles = 100000  # use 100K for 4hr training (744K total)
    havaids_val_split = 0.01       # 1% validation

    # === Model Architecture (RTX A6000 48GB) ===
    mode = "medium"  # tiny | small | medium | large

    MODEL_CONFIGS = {
        "tiny":  {"n_embd": 64,  "n_head": 4,  "n_layer": 3,  "batch_size": 32,  "max_iters": 2000,  "desc": "107K params, 5 min"},
        "small": {"n_embd": 128, "n_head": 4,  "n_layer": 4,  "batch_size": 64,  "max_iters": 5000,  "desc": "1.2M params, 20 min"},
        "medium":{"n_embd": 256, "n_head": 8,  "n_layer": 6,  "batch_size": 64,  "max_iters": 10000, "desc": "4.8M params, 1.5 hr"},
        "large": {"n_embd": 384, "n_head": 12, "n_layer": 8,  "batch_size": 32,  "max_iters": 15000, "desc": "16M params, 3 hr"},
    }

    _mc = MODEL_CONFIGS[mode]
    n_embd = _mc["n_embd"]
    n_head = _mc["n_head"]
    n_layer = _mc["n_layer"]
    batch_size = _mc["batch_size"]
    max_iters = _mc["max_iters"]

    # === Training ===
    block_size = 256
    eval_interval = 200
    eval_iters = 100
    learning_rate = 3e-4
    dropout = 0.2
    weight_decay = 1e-1
    warmup_iters = 100
    lr_decay_iters = max_iters
    min_lr = 1e-5
    grad_clip = 1.0

    # === Generation ===
    gen_max_tokens = 500
    gen_temperature = 0.8
    gen_top_k = 50

    # === Paths ===
    model_save_path = "./outputs/model_{}_{}.pth"
    bigram_save_path = "./outputs/bigram_baseline.pth"
    results_path = "./outputs/results.txt"
