class Config:
    # Hyperparameters (reduced for testing without GPU)
    batch_size = 4
    block_size = 32
    max_iters = 2
    eval_interval = 1
    eval_iters = 1
    learning_rate = 3e-4
    n_embd = 64
    n_head = 2
    n_layer = 2
    dropout = 0.2
    seed = 1337
    
    # Paths
    input_path = "/home/neo/Desktop/GITHUB MYZ21/GPT/data/data.txt"
    model_save_path = "/home/neo/Desktop/GITHUB MYZ21/GPT/outputs/model_{}.pth"
    
    # Device - force CPU for testing
    device = 'cpu'