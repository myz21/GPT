"""
Example script demonstrating how to use the character-level GPT model
"""

import torch
from model import GPTLanguageModel
from train import CharacterTokenizer, train, generate_text


def example_shakespeare():
    """Example using Shakespeare-like text"""
    
    # Sample Shakespeare text
    text = """
    First Citizen:
    Before we proceed any further, hear me speak.

    All:
    Speak, speak.

    First Citizen:
    You are all resolved rather to die than to famish?

    All:
    Resolved. Resolved.

    First Citizen:
    First, you know Caius Marcius is chief enemy to the people.

    All:
    We know't, we know't.

    First Citizen:
    Let us kill him, and we'll have corn at our own price.
    Is't a verdict?

    All:
    No more talking on't; let it be done: away, away!

    Second Citizen:
    One word, good citizens.

    First Citizen:
    We are accounted poor citizens, the patricians good.
    What authority surfeits on would relieve us: if they
    would yield us but the superfluity, while it were
    wholesome, we might guess they relieved us humanely; but
    they think we are too dear: the leanness that afflicts
    us, the object of our misery, is as an inventory to
    particularise their abundance; our sufferance is a gain
    to them Let us revenge this with our pikes, ere we
    become rakes: for the gods know I speak this in hunger
    for bread, not in thirst for revenge.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Train model
    print("Training on Shakespeare text...")
    model, tokenizer = train(
        text,
        n_embd=128,
        num_heads=4,
        num_layers=4,
        block_size=64,
        batch_size=16,
        max_iters=2000,
        learning_rate=1e-3,
        eval_interval=500,
        eval_iters=50,
        dropout=0.1,
        device=device
    )
    
    # Generate text with different prompts
    prompts = ["First", "Let us", "We are"]
    
    print("\n" + "=" * 70)
    print("GENERATED TEXT SAMPLES")
    print("=" * 70)
    
    for prompt in prompts:
        generated = generate_text(
            model, tokenizer, prompt, 
            max_new_tokens=150, 
            temperature=0.8,
            top_k=10,
            device=device
        )
        print(f"\nPrompt: '{prompt}'")
        print("-" * 70)
        print(generated)
        print("-" * 70)


def example_custom_text():
    """Example using custom text"""
    
    # You can use any text here
    custom_text = """
    The quick brown fox jumps over the lazy dog.
    Pack my box with five dozen liquor jugs.
    How vexingly quick daft zebras jump!
    The five boxing wizards jump quickly.
    Sphinx of black quartz, judge my vow.
    Two driven jocks help fax my big quiz.
    """
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    print("Training on custom text...")
    model, tokenizer = train(
        custom_text,
        n_embd=64,
        num_heads=4,
        num_layers=3,
        block_size=32,
        batch_size=8,
        max_iters=1000,
        learning_rate=1e-3,
        eval_interval=250,
        eval_iters=50,
        dropout=0.1,
        device=device
    )
    
    # Generate
    print("\nGenerating text...")
    generated = generate_text(
        model, tokenizer, "The", 
        max_new_tokens=100, 
        temperature=1.0,
        device=device
    )
    print(f"\nGenerated:\n{generated}")


if __name__ == '__main__':
    print("=" * 70)
    print("CHARACTER-LEVEL GPT - EXAMPLES")
    print("=" * 70)
    print("\nExample 1: Shakespeare-style text generation")
    print("=" * 70)
    
    example_shakespeare()
    
    print("\n\n" + "=" * 70)
    print("Example 2: Custom text generation")
    print("=" * 70)
    
    example_custom_text()
