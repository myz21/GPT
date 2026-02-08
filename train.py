"""
Training script for character-level GPT model
"""

import torch
import torch.nn as nn
from model import GPTLanguageModel


class CharacterTokenizer:
    """Simple character-level tokenizer"""
    
    def __init__(self, text):
        # Get all unique characters from the text
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        
        # Create mappings from characters to integers and vice versa
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
    
    def encode(self, text):
        """Convert text to a list of integers"""
        return [self.stoi[c] for c in text]
    
    def decode(self, indices):
        """Convert a list of integers back to text"""
        return ''.join([self.itos[i] for i in indices])


def get_batch(data, block_size, batch_size, device='cpu'):
    """Generate a small batch of data of inputs x and targets y"""
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device='cpu'):
    """Estimate loss on train and val sets"""
    out = {}
    model.eval()
    for split, data in [('train', train_data), ('val', val_data)]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, block_size, batch_size, device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def train(text, 
          n_embd=384,
          num_heads=6,
          num_layers=6,
          block_size=256,
          batch_size=64,
          max_iters=5000,
          learning_rate=3e-4,
          eval_interval=500,
          eval_iters=200,
          dropout=0.2,
          device='cpu'):
    """
    Train a character-level GPT model
    
    Args:
        text: Input text to train on
        n_embd: Embedding dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        block_size: Maximum context length
        batch_size: Batch size for training
        max_iters: Maximum number of training iterations
        learning_rate: Learning rate
        eval_interval: How often to evaluate
        eval_iters: Number of iterations for evaluation
        dropout: Dropout rate
        device: Device to train on ('cpu' or 'cuda')
    """
    
    # Initialize tokenizer
    tokenizer = CharacterTokenizer(text)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    print(f"Unique characters: {''.join(tokenizer.chars)}")
    
    # Encode the entire text
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    
    # Split into train and validation sets
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    
    # Initialize model
    model = GPTLanguageModel(
        vocab_size=tokenizer.vocab_size,
        n_embd=n_embd,
        num_heads=num_heads,
        num_layers=num_layers,
        block_size=block_size,
        dropout=dropout
    )
    model = model.to(device)
    
    # Print model size
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # Training loop
    for iter in range(max_iters):
        # Every eval_interval, evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(model, train_data, val_data, block_size, batch_size, eval_iters, device)
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        # Sample a batch of data
        xb, yb = get_batch(train_data, block_size, batch_size, device)
        
        # Evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    
    return model, tokenizer


def generate_text(model, tokenizer, prompt, max_new_tokens=500, temperature=1.0, top_k=None, device='cpu'):
    """Generate text from a trained model"""
    model.eval()
    
    # Encode the prompt
    context = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    
    # Generate
    generated = model.generate(context, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
    
    # Decode and return
    return tokenizer.decode(generated[0].tolist())


if __name__ == '__main__':
    # Example usage
    print("Character-level GPT Training")
    print("=" * 50)
    
    # Check if CUDA is available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Sample text (you can replace this with any text file)
    sample_text = """
    To be, or not to be, that is the question:
    Whether 'tis nobler in the mind to suffer
    The slings and arrows of outrageous fortune,
    Or to take arms against a sea of troubles
    And by opposing end them. To die—to sleep,
    No more; and by a sleep to say we end
    The heart-ache and the thousand natural shocks
    That flesh is heir to: 'tis a consummation
    Devoutly to be wish'd. To die, to sleep;
    To sleep, perchance to dream—ay, there's the rub:
    For in that sleep of death what dreams may come,
    When we have shuffled off this mortal coil,
    Must give us pause—there's the respect
    That makes calamity of so long life.
    """
    
    print(f"\nTraining on {len(sample_text)} characters...")
    print("\nStarting training...")
    
    # Train the model (using smaller parameters for quick testing)
    model, tokenizer = train(
        sample_text,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        block_size=64,
        batch_size=16,
        max_iters=1000,
        learning_rate=1e-3,
        eval_interval=200,
        eval_iters=50,
        dropout=0.1,
        device=device
    )
    
    print("\n" + "=" * 50)
    print("Generating text...")
    print("=" * 50)
    
    # Generate some text
    prompt = "To be"
    generated = generate_text(model, tokenizer, prompt, max_new_tokens=200, temperature=0.8, device=device)
    print(f"\nPrompt: '{prompt}'")
    print(f"\nGenerated text:\n{generated}")
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'chars': tokenizer.chars,
        'stoi': tokenizer.stoi,
        'itos': tokenizer.itos,
    }, 'gpt_model.pt')
    print("\n" + "=" * 50)
    print("Model saved to gpt_model.pt")
