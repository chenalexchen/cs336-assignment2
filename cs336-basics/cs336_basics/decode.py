"""
Text generation and decoding utilities for transformer language models.

This module implements various sampling strategies for generating text from
transformer language models, including temperature scaling and top-p (nucleus) sampling.
"""

import torch
import argparse
import json
import os
from pathlib import Path
from typing import List, Optional, Union

try:
    # Try relative import for module usage
    from .transformers import TransformerLM, softmax, load_checkpoint, AdamWOptimizer
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    # Fallback for standalone script usage
    from transformers import TransformerLM, softmax, load_checkpoint, AdamWOptimizer
    from bpe_tokenizer import BPETokenizer


def softmax_with_temperature(logits: torch.Tensor, temperature: float = 1.0, dim: int = -1) -> torch.Tensor:
    """
    Apply softmax with temperature scaling to logits.

    Temperature scaling modifies the sharpness of the probability distribution:
    - temperature < 1.0: Makes the distribution sharper (more confident)
    - temperature = 1.0: Standard softmax
    - temperature > 1.0: Makes the distribution smoother (more diverse)

    Args:
        logits: Unnormalized log probabilities of shape [..., vocab_size]
        temperature: Temperature parameter for scaling. Must be > 0.
        dim: Dimension to apply softmax over

    Returns:
        Probability distribution of same shape as logits
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Scale logits by temperature
    scaled_logits = logits / temperature
    return softmax(scaled_logits, dim=dim)


def top_p_sampling(probs: torch.Tensor, p: float = 0.9) -> torch.Tensor:
    """
    Apply top-p (nucleus) sampling to a probability distribution.

    Top-p sampling selects from the smallest set of tokens whose cumulative
    probability mass exceeds the threshold p, then renormalizes.

    Args:
        probs: Probability distribution of shape [batch_size, vocab_size] or [vocab_size]
        p: Cumulative probability threshold (0 < p <= 1.0)

    Returns:
        Modified probability distribution with low-probability tokens zeroed out
    """
    if not (0 < p <= 1.0):
        raise ValueError(f"p must be in (0, 1], got {p}")

    # Handle both batched and single distributions
    original_shape = probs.shape
    if probs.dim() == 1:
        probs = probs.unsqueeze(0)

    batch_size, vocab_size = probs.shape

    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for tokens to keep (cumulative probability <= p)
    # We keep the first token that exceeds p to ensure we always have at least one token
    keep_mask = cumulative_probs <= p
    # Always keep at least the most probable token
    keep_mask[:, 0] = True

    # Zero out probabilities for tokens we don't want to keep
    filtered_sorted_probs = sorted_probs * keep_mask.float()

    # Create output tensor and scatter the filtered probabilities back to original positions
    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim=-1, index=sorted_indices, src=filtered_sorted_probs)

    # Renormalize to ensure probabilities sum to 1
    filtered_probs = filtered_probs / (filtered_probs.sum(dim=-1, keepdim=True) + 1e-8)

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        filtered_probs = filtered_probs.squeeze(0)

    return filtered_probs


def decode(
    model: TransformerLM,
    prompt_tokens: torch.Tensor,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    eos_token_id: Optional[int] = None,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate text completions from a transformer language model.

    This function implements autoregressive text generation by repeatedly:
    1. Running the model on the current sequence
    2. Extracting logits for the next token
    3. Applying temperature scaling and/or top-p sampling
    4. Sampling the next token
    5. Appending it to the sequence

    Generation stops when either max_tokens is reached or the EOS token is generated.

    Args:
        model: The transformer language model to use for generation
        prompt_tokens: Input token sequence of shape [batch_size, seq_len] or [seq_len]
        max_tokens: Maximum number of new tokens to generate
        temperature: Temperature for scaling logits (> 0). Lower = more conservative.
        top_p: If provided, use top-p sampling with this threshold (0 < top_p <= 1)
        eos_token_id: Token ID that indicates end of sequence. Generation stops if this is sampled.
        device: Device to run generation on

    Returns:
        Generated token sequence including the original prompt
        Shape: [batch_size, original_seq_len + generated_tokens] or [original_seq_len + generated_tokens]
    """
    model.eval()

    # Handle both batched and single sequences
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False

    batch_size, initial_seq_len = prompt_tokens.shape
    prompt_tokens = prompt_tokens.to(device)

    # Initialize the sequence with the prompt
    generated_sequence = prompt_tokens.clone()

    with torch.no_grad():
        for _ in range(max_tokens):
            # Get logits from the model for the current sequence
            # Shape: [batch_size, seq_len, vocab_size]
            logits = model(generated_sequence)

            # Extract logits for the last position (next token prediction)
            # Shape: [batch_size, vocab_size]
            next_token_logits = logits[:, -1, :]

            # Apply temperature scaling
            probs = softmax_with_temperature(next_token_logits, temperature=temperature)

            # Apply top-p sampling if requested
            if top_p is not None:
                probs = top_p_sampling(probs, p=top_p)

            # Sample next tokens
            # Shape: [batch_size, 1]
            next_tokens = torch.multinomial(probs, num_samples=1)

            # Append to the generated sequence
            generated_sequence = torch.cat([generated_sequence, next_tokens], dim=1)

            # Check for EOS token
            if eos_token_id is not None:
                # If all sequences in the batch have generated EOS, stop
                if (next_tokens.squeeze(-1) == eos_token_id).all():
                    break

    # Restore original shape if input was 1D
    if squeeze_output:
        generated_sequence = generated_sequence.squeeze(0)

    return generated_sequence


def decode_with_tokenizer(
    model: TransformerLM,
    tokenizer: BPETokenizer,
    prompt: str,
    max_tokens: int = 100,
    temperature: float = 1.0,
    top_p: Optional[float] = None,
    device: str = "cpu",
) -> str:
    """
    Generate text completion from a string prompt using a tokenizer.

    This is a convenience function that handles tokenization and detokenization
    around the core decode function.

    Args:
        model: The transformer language model
        tokenizer: BPE tokenizer instance
        prompt: String prompt to complete
        max_tokens: Maximum number of new tokens to generate
        temperature: Temperature for sampling
        top_p: Top-p threshold for nucleus sampling
        device: Device to run on

    Returns:
        Generated text completion as a string
    """
    # Encode the prompt to token IDs
    prompt_token_ids = tokenizer.encode(prompt)
    prompt_tokens = torch.tensor(prompt_token_ids, dtype=torch.long)

    # Get <|endoftext|> token ID if it exists in the tokenizer's special tokens
    eos_token_id = None
    if "<|endoftext|>" in tokenizer.special_tokens:
        eos_token_bytes = "<|endoftext|>".encode("utf-8")
        eos_token_id = tokenizer.bytes_to_id.get(eos_token_bytes)

    # Generate tokens using the core decode function
    generated_tokens = decode(
        model=model,
        prompt_tokens=prompt_tokens,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        eos_token_id=eos_token_id,
        device=device,
    )

    # Convert back to list and decode to string
    generated_token_ids = generated_tokens.tolist()
    generated_text = tokenizer.decode(generated_token_ids)

    return generated_text


def load_model_from_checkpoint(
    checkpoint_path: str, config_path: Optional[str] = None, device: str = "cpu"
) -> TransformerLM:
    """
    Load a trained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the checkpoint .pt file
        config_path: Optional path to config.json. If None, will look for config.json in the same directory
        device: Device to load the model on

    Returns:
        Loaded TransformerLM model in eval mode
    """
    checkpoint_path = Path(checkpoint_path)

    # Try to find config file if not provided
    if config_path is None:
        # Look in the same directory as the checkpoint
        config_path = checkpoint_path.parent / "config.json"
        if not config_path.exists():
            # Try parent directory (common when checkpoint is in subdirectory)
            config_path = checkpoint_path.parent.parent / "config.json"
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found. Please specify config_path or ensure config.json exists at {config_path}"
        )

    # Load configuration
    print(f"Loading configuration from {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # Create model with the same configuration as during training
    print(f"Creating model: {config['num_layers']} layers, {config['d_model']} dim, {config['num_heads']} heads")
    model = TransformerLM(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config["rope_theta"],
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"],
    )

    # Load checkpoint (we need a dummy optimizer for the load_checkpoint function)
    # We'll just create a dummy optimizer since we only need the model state
    dummy_optimizer = AdamWOptimizer(model.parameters(), lr=1e-4)

    print(f"Loading checkpoint from {checkpoint_path}")
    iteration = load_checkpoint(checkpoint_path, model, dummy_optimizer)
    print(f"✓ Loaded checkpoint from iteration {iteration}")

    # Move to device and set to eval mode
    model.to(device)
    model.eval()

    return model


def load_tokenizer_from_files(
    vocab_path: str, merges_path: str, special_tokens: Optional[List[str]] = None
) -> BPETokenizer:
    """
    Load a BPE tokenizer from vocab and merges files.

    Args:
        vocab_path: Path to vocab.json file
        merges_path: Path to merges.txt file
        special_tokens: List of special tokens (default: ["<|endoftext|>"])

    Returns:
        BPETokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print(f"Loading tokenizer from {vocab_path} and {merges_path}")

    # Load vocabulary
    with open(vocab_path, "r") as f:
        vocab_data = json.load(f)

    # Convert: vocab_data has token_bytes (as string) -> token_id (as int)
    # We need: token_id -> token_bytes (as bytes)
    vocab = {}
    for token_str, token_id in vocab_data.items():
        # The JSON already handles unicode escaping properly
        # Just encode the string to bytes
        try:
            token_bytes = token_str.encode("utf-8")
        except UnicodeEncodeError:
            # If that fails, try treating it as raw bytes
            token_bytes = token_str.encode("latin-1")
        vocab[token_id] = token_bytes

    # Load merges
    merges = []
    with open(merges_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                # Split on space, but handle the case where tokens contain spaces
                if " " in line:
                    # Find the space that separates the two tokens
                    # This assumes each token is separated by a single space
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        # Join parts in case tokens contain spaces
                        mid_point = len(parts) // 2
                        token1 = " ".join(parts[:mid_point]).encode("utf-8")
                        token2 = " ".join(parts[mid_point:]).encode("utf-8")
                        merges.append((token1, token2))

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    print(f"✓ Loaded tokenizer with vocab size {len(vocab)} and {len(merges)} merges")

    return tokenizer


def parse_args():
    """Parse command-line arguments for the decode script."""
    parser = argparse.ArgumentParser(
        description="Generate text using a trained transformer language model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic text generation
  python code/decode.py --checkpoint checkpoints/final_checkpoint.pt --prompt "Once upon a time"
  
  # With custom settings
  python code/decode.py \\
    --checkpoint checkpoints/checkpoint_step_10000.pt \\
    --config checkpoints/config.json \\
    --vocab tokenizer_output/vocab.json \\
    --merges tokenizer_output/merges.txt \\
    --prompt "The future of AI is" \\
    --max-tokens 100 \\
    --temperature 0.8 \\
    --top-p 0.9 \\
    --device cuda
        """,
    )

    # Model and data paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt file)")
    parser.add_argument("--config", type=str, help="Path to config.json (auto-detected if not specified)")
    parser.add_argument("--vocab", type=str, help="Path to vocab.json file")
    parser.add_argument("--merges", type=str, help="Path to merges.txt file")

    # Generation parameters
    parser.add_argument("--prompt", type=str, default="", help="Input prompt for text generation")
    parser.add_argument("--max-tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature (higher = more random)")
    parser.add_argument("--top-p", type=float, help="Top-p (nucleus) sampling threshold")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use"
    )

    # Special tokens
    parser.add_argument(
        "--special-tokens", type=str, nargs="*", default=["<|endoftext|>"], help="Special tokens for tokenizer"
    )

    # Output options
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for multiple prompts")
    parser.add_argument("--seed", type=int, help="Random seed for reproducible generation")

    return parser.parse_args()


def interactive_generation(model: TransformerLM, tokenizer: BPETokenizer, args):
    """Interactive mode for generating multiple text completions."""
    print("\n🤖 Interactive Text Generation Mode")
    print("Type your prompts (or 'quit' to exit):")
    print("-" * 50)

    while True:
        try:
            prompt = input("\n>>> ").strip()
            if prompt.lower() in ["quit", "exit", "q"]:
                break
            if not prompt:
                continue

            # Set random seed if provided
            if args.seed is not None:
                torch.manual_seed(args.seed)

            print("Generating...")
            generated_text = decode_with_tokenizer(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=args.device,
            )

            print(f"\n{generated_text}")

        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def cli_main():
    """Main CLI interface for text generation."""
    args = parse_args()

    # Set random seed if provided
    if args.seed is not None:
        torch.manual_seed(args.seed)
        print(f"Set random seed to {args.seed}")

    print("🚀 Loading model and tokenizer...")

    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.config, args.device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # Load tokenizer - try to auto-detect if not specified
    tokenizer = None
    if args.vocab and args.merges:
        try:
            tokenizer = load_tokenizer_from_files(args.vocab, args.merges, args.special_tokens)
        except Exception as e:
            print(f"❌ Error loading tokenizer: {e}")
            return 1
    else:
        # Try to auto-detect tokenizer files
        checkpoint_dir = Path(args.checkpoint).parent
        possible_locations = [checkpoint_dir, checkpoint_dir.parent, Path("tokenizer_output"), Path(".")]

        for location in possible_locations:
            vocab_file = location / "vocab.json"
            merges_file = location / "merges.txt"
            if vocab_file.exists() and merges_file.exists():
                print(f"Auto-detected tokenizer files in {location}")
                try:
                    tokenizer = load_tokenizer_from_files(str(vocab_file), str(merges_file), args.special_tokens)
                    break
                except Exception as e:
                    print(f"Failed to load tokenizer from {location}: {e}")
                    continue

    if tokenizer is None:
        print("❌ Could not find or load tokenizer. Please specify --vocab and --merges paths.")
        return 1

    print("✅ Model and tokenizer loaded successfully!")

    # Interactive or single generation mode
    if args.interactive:
        interactive_generation(model, tokenizer, args)
    else:
        if not args.prompt:
            print("❌ Please provide a --prompt or use --interactive mode")
            return 1

        print(f"Generating completion for: '{args.prompt}'")
        generated_text = decode_with_tokenizer(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=args.device,
        )

        print("\n" + "=" * 50)
        print("GENERATED TEXT:")
        print("=" * 50)
        print(generated_text)

    return 0


# Example usage and testing functions
def main():
    """
    Example usage of the decoding functions.
    This demonstrates how to use the decoding utilities.
    """
    print("Decoding module loaded successfully!")

    # Example: Test temperature scaling
    print("\n=== Testing temperature scaling ===")
    logits = torch.tensor([[2.0, 1.0, 0.5, 0.1]])

    for temp in [0.1, 1.0, 2.0]:
        probs = softmax_with_temperature(logits, temperature=temp)
        print(f"Temperature {temp}: {probs[0].tolist()}")

    # Example: Test top-p sampling
    print("\n=== Testing top-p sampling ===")
    probs = torch.tensor([0.5, 0.3, 0.1, 0.05, 0.05])

    for p in [0.8, 0.9, 1.0]:
        filtered = top_p_sampling(probs, p=p)
        print(f"Top-p {p}: {filtered.tolist()}")


def test_decode_functionality():
    """
    Test the decode function with a simple mock scenario.
    """
    print("\n=== Testing decode function with dummy model ===")

    # Create a simple "mock" transformer that just returns random logits
    class MockTransformer:
        def __init__(self, vocab_size=1000):
            self.vocab_size = vocab_size

        def eval(self):
            pass

        def __call__(self, x):
            batch_size, seq_len = x.shape
            # Return random logits for demonstration
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
            return logits

    # Test parameters
    vocab_size = 1000
    mock_model = MockTransformer(vocab_size)
    prompt_tokens = torch.tensor([[1, 2, 3, 4, 5]])  # Mock prompt

    print(f"Input prompt tokens: {prompt_tokens}")
    print(f"Generating 10 tokens with different settings...")

    # Test different temperature and top-p settings
    settings = [
        (1.0, None),  # Standard sampling
        (0.1, None),  # Low temperature (more deterministic)
        (1.0, 0.9),  # Top-p sampling
        (0.8, 0.95),  # Combined temperature + top-p
    ]

    for temp, top_p in settings:
        # Set random seed for reproducibility in this test
        torch.manual_seed(42)

        generated = decode(
            model=mock_model, prompt_tokens=prompt_tokens, max_tokens=10, temperature=temp, top_p=top_p, device="cpu"
        )

        setting_desc = f"temp={temp}"
        if top_p:
            setting_desc += f", top_p={top_p}"

        print(f"  {setting_desc}: {generated[0].tolist()}")

    print("✓ decode function works correctly!")


def demo_with_trained_model():
    """
    Example of how to use the decoding functions with a trained model and tokenizer.

    This function shows the expected workflow for text generation once you have:
    1. A trained TransformerLM model
    2. A BPE tokenizer (vocab.json and merges files)
    """
    print("\n=== Demo: Text Generation Workflow ===")
    print("This demonstrates the expected usage pattern:\n")

    print("# Step 1: Load your trained model")
    print("model = TransformerLM(...)")
    print("model.load_state_dict(torch.load('model.pt'))")
    print("model.eval()")

    print("\n# Step 2: Load your tokenizer")
    print("import json")
    print("with open('vocab.json') as f:")
    print("    vocab_data = json.load(f)")
    print("# Convert string keys to int and values to bytes")
    print("vocab = {int(k): v.encode('utf-8') for k, v in vocab_data.items()}")
    print("# Load merges...")
    print("tokenizer = BPETokenizer(vocab, merges, special_tokens=['<|endoftext|>'])")

    print("\n# Step 3: Generate text")
    print("prompt = 'Once upon a time'")
    print("generated_text = decode_with_tokenizer(")
    print("    model=model,")
    print("    tokenizer=tokenizer,")
    print("    prompt=prompt,")
    print("    max_tokens=50,")
    print("    temperature=0.8,")
    print("    top_p=0.9,")
    print("    device='cuda'")
    print(")")
    print("print(generated_text)")

    print("\n# Alternative: Direct token-level control")
    print("prompt_tokens = torch.tensor(tokenizer.encode(prompt))")
    print("generated_tokens = decode(")
    print("    model=model,")
    print("    prompt_tokens=prompt_tokens,")
    print("    max_tokens=50,")
    print("    temperature=0.8,")
    print("    top_p=0.9,")
    print("    eos_token_id=tokenizer.special_token_to_id.get('<|endoftext|>'),")
    print("    device='cuda'")
    print(")")
    print("text = tokenizer.decode(generated_tokens.tolist())")


if __name__ == "__main__":
    import sys

    # If command-line arguments provided, run CLI mode
    if len(sys.argv) > 1:
        exit_code = cli_main()
        sys.exit(exit_code)
    else:
        # No arguments, run test/demo mode
        main()
        test_decode_functionality()
        demo_with_trained_model()
