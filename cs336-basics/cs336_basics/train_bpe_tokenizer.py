#!/usr/bin/env python3
"""
CLI script for training BPE tokenizers.

This script provides a command-line interface for training Byte Pair Encoding (BPE)
tokenizers from text data. It saves the resulting vocabulary and merges files for later use.
"""

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

try:
    # Try relative import for module usage
    from .bpe_tokenizer import (
        train_bpe,
        BPETokenizer,
        extract_word_freqs,
        train_bpe_from_freqs,
        save_word_freqs,
        load_word_freqs,
    )
except ImportError:
    # Fallback for standalone script usage
    from bpe_tokenizer import (
        train_bpe,
        BPETokenizer,
        extract_word_freqs,
        train_bpe_from_freqs,
        save_word_freqs,
        load_word_freqs,
    )


def save_word_freqs(word_freqs: dict, output_file: str):
    """
    Save word frequencies to a JSON file, ordered by frequency (descending).

    Args:
        word_freqs: Dictionary mapping word tuples to frequencies
        output_file: Path to output JSON file
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to list and sort by frequency (descending), then by word for determinism
    sorted_freqs = sorted(word_freqs.items(), key=lambda x: (-x[1], x[0]))

    # Convert to JSON-serializable format
    json_data = []
    for word_tuple, freq in sorted_freqs:
        # Convert tuple of bytes to string representation for JSON
        word_str = "".join(chr(b) if b < 128 else f"\\x{b:02x}" for b in word_tuple)
        json_data.append(
            {
                "word": list(word_tuple),  # Store as list of integers
                "word_display": word_str,  # Human-readable representation
                "frequency": freq,
            }
        )

    with open(output_file, "w") as f:
        json.dump(json_data, f, indent=2)

    print(f"✓ Saved {len(word_freqs)} word frequencies to {output_file}")


def load_word_freqs(input_file: str) -> dict:
    """
    Load word frequencies from a JSON file.

    Args:
        input_file: Path to JSON file containing word frequencies

    Returns:
        Dictionary mapping word tuples to frequencies
    """
    with open(input_file, "r") as f:
        json_data = json.load(f)

    word_freqs = {}
    for item in json_data:
        word_tuple = tuple(item["word"])
        frequency = item["frequency"]
        word_freqs[word_tuple] = frequency

    print(f"✅ Loaded {len(word_freqs)} word frequencies from {input_file}")
    return word_freqs


def save_tokenizer_files(vocab: dict, merges: list, output_dir: str, special_tokens: List[str]):
    """
    Save tokenizer vocabulary and merges to files.

    Args:
        vocab: Vocabulary dictionary mapping token IDs to bytes
        merges: List of merge rules as (token1, token2) tuples
        output_dir: Output directory path
        special_tokens: List of special tokens used during training
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Saving tokenizer files to {output_path}")

    # Save vocabulary as JSON (convert bytes to strings)
    vocab_path = output_path / "vocab.json"
    vocab_for_json = {}
    for k, v in vocab.items():
        try:
            # Try to decode as UTF-8 first
            token_str = v.decode("utf-8")
        except UnicodeDecodeError:
            # If that fails, decode as latin-1 (preserves all byte values)
            token_str = v.decode("latin-1")
        vocab_for_json[token_str] = k

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_for_json, f, indent=2, ensure_ascii=False)

    print(f"✓ Saved vocabulary to {vocab_path}")

    # Save merges as text file
    merges_path = output_path / "merges.txt"
    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("#version: 0.2\n")
        for token1, token2 in merges:
            # Convert bytes back to strings for saving
            try:
                token1_str = token1.decode("utf-8")
            except UnicodeDecodeError:
                token1_str = token1.decode("latin-1")
            try:
                token2_str = token2.decode("utf-8")
            except UnicodeDecodeError:
                token2_str = token2.decode("latin-1")
            f.write(f"{token1_str} {token2_str}\n")

    print(f"✓ Saved merges to {merges_path}")

    # Save training statistics
    stats_path = output_path / "training_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Vocabulary size: {len(vocab)}\n")
        f.write(f"Number of merges: {len(merges)}\n")
        f.write(f"Special tokens: {special_tokens}\n")
        f.write(f"Base tokens (0-255): {sum(1 for k in vocab.keys() if k < 256)}\n")
        f.write(f"Special tokens: {sum(1 for k in vocab.keys() if 256 <= k < 256 + len(special_tokens))}\n")
        f.write(f"Learned merges: {sum(1 for k in vocab.keys() if k >= 256 + len(special_tokens))}\n")

    print(f"✓ Saved training stats to {stats_path}")


def parse_args():
    """Parse command-line arguments for BPE training."""
    parser = argparse.ArgumentParser(
        description="Train a BPE (Byte Pair Encoding) tokenizer from text data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic BPE training
  python code/train_bpe_tokenizer.py \\
    --input data/train.txt \\
    --output tokenizer_output \\
    --vocab-size 32000
  
  # With custom special tokens
  python code/train_bpe_tokenizer.py \\
    --input data/corpus.txt \\
    --output my_tokenizer \\
    --vocab-size 50000 \\
    --special-tokens "<|endoftext|>" "<|pad|>" "<|unk|>"
  
  # Training for code
  python code/train_bpe_tokenizer.py \\
    --input code_corpus.txt \\
    --output code_tokenizer \\
    --vocab-size 32000 \\
    --special-tokens "<|endoftext|>" "<|fim_prefix|>" "<|fim_middle|>" "<|fim_suffix|>"
        """,
    )

    # Required arguments
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input text file for training")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory for tokenizer files")
    parser.add_argument(
        "--vocab-size",
        "-v",
        type=int,
        help="Target vocabulary size (includes base bytes and special tokens, required for full and train-from-freqs modes)",
    )

    # Optional arguments
    parser.add_argument(
        "--special-tokens",
        "-s",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens to add to vocabulary (default: ['<|endoftext|>'])",
    )

    # Mode options
    parser.add_argument(
        "--mode",
        choices=["full", "extract-freqs", "train-from-freqs"],
        default="full",
        help="Training mode: 'full' (extract freqs + train), 'extract-freqs' (only extract word frequencies), 'train-from-freqs' (load freqs + train)",
    )
    parser.add_argument(
        "--word-freqs-file", type=str, help="Path to word frequencies JSON file (required for 'train-from-freqs' mode)"
    )

    # Output options
    parser.add_argument("--test-text", type=str, help="Optional test text to encode/decode after training")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output during training")

    return parser.parse_args()


def test_tokenizer(vocab: dict, merges: list, special_tokens: List[str], test_text: str):
    """Test the trained tokenizer with sample text."""
    print(f"\n=== Testing Tokenizer ===")
    print(f'Test text: "{test_text}"')

    # Create tokenizer instance
    tokenizer = BPETokenizer(vocab, merges, special_tokens)

    # Test encoding
    token_ids = tokenizer.encode(test_text)
    print(f"Token IDs: {token_ids}")
    print(f"Number of tokens: {len(token_ids)}")

    # Test decoding
    decoded_text = tokenizer.decode(token_ids)
    print(f'Decoded text: "{decoded_text}"')

    # Check if encoding/decoding is lossless
    if decoded_text == test_text:
        print("✅ Lossless encoding/decoding!")
    else:
        print("⚠️  Encoding/decoding mismatch:")
        print(f"   Original: {repr(test_text)}")
        print(f"   Decoded:  {repr(decoded_text)}")

    # Show some token mappings
    print(f"\nToken breakdown:")
    for i, token_id in enumerate(token_ids[:10]):  # Show first 10 tokens
        token_bytes = vocab[token_id]
        token_str = token_bytes.decode("utf-8", errors="replace")
        print(f"  {i}: {token_id} -> {repr(token_str)}")
        if i >= 9 and len(token_ids) > 10:
            print(f"  ... ({len(token_ids) - 10} more tokens)")
            break


def main():
    """Main function for BPE tokenizer training."""
    args = parse_args()

    print("🚀 Starting BPE tokenizer training")
    print("=" * 50)

    # Validate arguments based on mode
    if args.mode == "train-from-freqs":
        if not args.word_freqs_file:
            print("❌ Error: --word-freqs-file is required for 'train-from-freqs' mode")
            return 1
        if not os.path.exists(args.word_freqs_file):
            print(f"❌ Error: Word frequencies file '{args.word_freqs_file}' not found")
            return 1
    else:
        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found")
            return 1

    # Validate vocab_size for modes that need it
    if args.mode != "extract-freqs":
        if not args.vocab_size:
            print(f"❌ Error: --vocab-size is required for '{args.mode}' mode")
            return 1
        if args.vocab_size <= 256:
            print(f"❌ Error: vocab_size ({args.vocab_size}) must be > 256 (base byte vocabulary)")
            return 1

    # Show configuration
    print(f"🔧 Mode: {args.mode}")
    if args.mode != "train-from-freqs":
        print(f"📁 Input file: {args.input}")
        file_size = os.path.getsize(args.input)
        print(f"📏 Input file size: {file_size:,} bytes ({file_size / 1024 / 1024:.1f} MB)")
    if args.mode != "extract-freqs":
        print(f"📁 Output directory: {args.output}")
        print(f"📊 Target vocabulary size: {args.vocab_size:,}")
    if args.word_freqs_file:
        print(f"📋 Word frequencies file: {args.word_freqs_file}")
    print(f"🏷️  Special tokens: {args.special_tokens}")

    try:
        start_time = time.time()

        if args.mode == "extract-freqs":
            # Only extract word frequencies
            print(f"\n📊 Extracting word frequencies...")
            word_freqs = extract_word_freqs(args.input, args.special_tokens)

            # Save word frequencies
            freqs_output = f"{args.output}/word_freqs.json"
            save_word_freqs(word_freqs, freqs_output)

            end_time = time.time()
            print(f"✅ Word frequency extraction completed in {end_time - start_time:.1f}s")
            print(f"📊 Extracted {len(word_freqs)} unique word types")
            print(f"📁 Word frequencies saved to: {freqs_output}")
            return 0

        elif args.mode == "train-from-freqs":
            # Load word frequencies and train
            print(f"\n📋 Loading word frequencies from {args.word_freqs_file}...")
            word_freqs = load_word_freqs(args.word_freqs_file)

            print(f"\n⏳ Training BPE from loaded frequencies...")
            vocab, merges = train_bpe_from_freqs(word_freqs, args.vocab_size, args.special_tokens)

        else:  # full mode
            # Full training pipeline
            print(f"\n⏳ Training BPE tokenizer...")

            if args.verbose:
                print("Training in verbose mode...")

            vocab, merges = train_bpe(
                input_path=args.input, vocab_size=args.vocab_size, special_tokens=args.special_tokens
            )

        end_time = time.time()
        training_time = end_time - start_time

        print(f"✅ Training completed in {training_time:.1f}s")
        print(f"📈 Final vocabulary size: {len(vocab):,}")
        print(f"🔗 Number of merges learned: {len(merges):,}")

        # Calculate effective compression
        base_tokens = sum(1 for k in vocab.keys() if k < 256)
        special_count = len(args.special_tokens)
        learned_merges = len(vocab) - base_tokens - special_count

        print(f"📊 Token breakdown:")
        print(f"   • Base bytes (0-255): {base_tokens}")
        print(f"   • Special tokens: {special_count}")
        print(f"   • Learned merge tokens: {learned_merges}")

        # Save tokenizer files
        save_tokenizer_files(vocab, merges, args.output, args.special_tokens)

        # Test tokenizer if test text provided
        if args.test_text:
            test_tokenizer(vocab, merges, args.special_tokens, args.test_text)
        else:
            # Default test with a simple sentence
            default_test = "Hello world! This is a test of the BPE tokenizer."
            test_tokenizer(vocab, merges, args.special_tokens, default_test)

        print(f"\n🎉 BPE tokenizer training completed successfully!")
        print(f"📁 Tokenizer files saved to: {args.output}")
        print(f"\nTo use this tokenizer:")
        print(
            f"  python code/bpe_tokenize.py --vocab {args.output}/vocab.json --merges {args.output}/merges.txt --input your_text.txt"
        )

        return 0

    except Exception as e:
        print(f"❌ Error during training: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
