#!/usr/bin/env python3
"""
Text generation script using a trained transformer model.

This script loads a trained transformer model and generates text samples
using various generation strategies like greedy decoding, sampling, etc.
"""

import argparse
import json
import torch
import torch.nn.functional as F
from pathlib import Path
import sys
import os

# Add the current directory to Python path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from transformers import TransformerLM
    from bpe_tokenizer import BPETokenizer
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Make sure transformers.py and bpe_tokenizer.py are in the same directory")
    sys.exit(1)


def load_tokenizer(vocab_path: str, merges_path: str):
    """Load the BPE tokenizer."""
    print(f"Loading tokenizer from {vocab_path} and {merges_path}")
    
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    
    # Convert vocab format
    vocab = {}
    for token_str, token_id in vocab_data.items():
        try:
            token_bytes = token_str.encode("utf-8")
            vocab[token_id] = token_bytes
        except UnicodeDecodeError:
            # Handle special tokens that might not be valid UTF-8
            vocab[token_id] = token_str.encode("utf-8", errors="replace")
    
    # Load merges
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                try:
                    pair = tuple(line.split())
                    if len(pair) == 2:
                        merges.append(pair)
                except:
                    continue
    
    # Create tokenizer
    tokenizer = BPETokenizer(vocab, merges, special_tokens=["<|endoftext|>"])
    print(f"✓ Loaded tokenizer: {len(vocab)} vocab, {len(merges)} merges")
    return tokenizer


def load_model_and_config(checkpoint_path: str, config_path: str, device: str = "cuda"):
    """Load the trained model and its configuration."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load configuration
    with open(config_path, "r") as f:
        config = json.load(f)
    
    print(f"Model config: {config}")
    
    # Create model
    model = TransformerLM(
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        theta=config.get("rope_theta", 10000.0),
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        num_layers=config["num_layers"]
    )
    
    # Store context length for generation
    model.context_length = config["context_length"]
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
        print(f"✓ Loaded checkpoint from step {checkpoint.get('step', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print(f"✓ Loaded model weights")
    
    # Move model to device
    model = model.to(device)
    model.eval()
    return model, config


def generate_text(
    model, 
    tokenizer, 
    prompt: str = "", 
    max_length: int = 100, 
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.9,
    device: str = "cuda"
):
    """Generate text using the trained model."""
    
    # Tokenize prompt
    if prompt:
        input_ids = tokenizer.encode(prompt)
        print(f"Prompt tokens: {input_ids}")
    else:
        # Start with endoftext token
        try:
            endoftext_tokens = tokenizer.encode("<|endoftext|>")
            input_ids = endoftext_tokens[:1]  # Just the first token
        except:
            input_ids = [0]  # Fallback to first token
    
    input_ids = torch.tensor(input_ids, device=device).unsqueeze(0)
    print(f"Starting generation with {input_ids.shape[1]} tokens...")
    
    generated_ids = input_ids.clone()
    
    with torch.no_grad():
        for i in range(max_length):
            # Get model predictions
            logits = model(generated_ids)  # [1, seq_len, vocab_size]
            
            # Get last token logits
            next_token_logits = logits[0, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k > 0:
                top_k_vals, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                next_token_logits = torch.where(
                    next_token_logits < top_k_vals[-1],
                    torch.full_like(next_token_logits, float('-inf')),
                    next_token_logits
                )
            
            # Apply top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = 0
                
                # Create a mask instead of indexing
                mask = torch.zeros_like(next_token_logits, dtype=torch.bool)
                mask.scatter_(0, sorted_indices[sorted_indices_to_remove], True)
                next_token_logits = torch.where(mask, torch.full_like(next_token_logits, float('-inf')), next_token_logits)
            
            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Add to sequence
            generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=1)
            
            # Check for end of text token or max context length
            endoftext_id = None
            if hasattr(tokenizer, 'special_token_ids'):
                endoftext_id = tokenizer.special_token_ids.get("<|endoftext|>", None)
            elif hasattr(tokenizer, 'encode'):
                # Try to encode the special token
                try:
                    endoftext_tokens = tokenizer.encode("<|endoftext|>")
                    if len(endoftext_tokens) == 1:
                        endoftext_id = endoftext_tokens[0]
                except:
                    pass
                    
            if endoftext_id is not None and next_token.item() == endoftext_id:
                break
                
            # Truncate if we exceed context length
            if generated_ids.shape[1] > model.context_length:
                generated_ids = generated_ids[:, -model.context_length:]
    
    # Decode generated text
    generated_tokens = generated_ids[0].cpu().tolist()
    generated_text = tokenizer.decode(generated_tokens)
    
    return generated_text, generated_tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text using a trained transformer model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--config", type=str, required=True, help="Path to model config")
    parser.add_argument("--vocab", type=str, required=True, help="Path to vocab.json")
    parser.add_argument("--merges", type=str, required=True, help="Path to merges.txt")
    parser.add_argument("--prompt", type=str, default="", help="Text prompt to start generation")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k sampling")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p (nucleus) sampling")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of samples to generate")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use")
    
    args = parser.parse_args()
    
    print("🚀 Loading model and tokenizer...")
    
    # Load tokenizer
    tokenizer = load_tokenizer(args.vocab, args.merges)
    
    # Load model
    model, config = load_model_and_config(args.checkpoint, args.config, args.device)
    
    print(f"✓ Model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    print()
    
    # Generate samples
    print(f"🎯 Generating {args.num_samples} samples...")
    print(f"Settings: temp={args.temperature}, top_k={args.top_k}, top_p={args.top_p}, max_len={args.max_length}")
    print("=" * 80)
    
    for i in range(args.num_samples):
        print(f"\n📝 Sample {i+1}:")
        print("-" * 40)
        
        try:
            generated_text, tokens = generate_text(
                model, tokenizer, 
                prompt=args.prompt,
                max_length=args.max_length,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                device=args.device
            )
            
            print(f"Generated text ({len(tokens)} tokens):")
            print(generated_text)
            print()
            
        except Exception as e:
            print(f"Error generating sample {i+1}: {e}")
            continue
    
    print("=" * 80)
    print("✅ Generation complete!")


if __name__ == "__main__":
    main()