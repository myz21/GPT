class Config:
    # Hyperparameters
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    eval_iters = 50
    learning_rate = 3e-4
    n_embd = 256
    n_head = 6
    n_layer = 6
    dropout = 0.2
    seed = 1337
    
    # Paths
    input_path = "./data/data.txt"
    model_save_path = "./outputs/model_{}.pth"
    
    # Device
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'