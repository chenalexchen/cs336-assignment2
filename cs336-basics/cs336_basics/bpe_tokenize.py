#!/usr/bin/env python3
"""
CLI script for tokenizing text using trained BPE tokenizers.

This script provides a command-line interface for tokenizing and detokenizing text
using BPE tokenizers trained with train_bpe_tokenizer.py.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

try:
    # Try relative import for module usage
    from .bpe_tokenizer import BPETokenizer
except ImportError:
    # Fallback for standalone script usage
    from bpe_tokenizer import BPETokenizer


def load_tokenizer(vocab_path: str, merges_path: str, special_tokens: Optional[List[str]] = None) -> BPETokenizer:
    """
    Load a BPE tokenizer from vocab and merges files.

    Args:
        vocab_path: Path to vocab.json file
        merges_path: Path to merges.txt file
        special_tokens: Optional list of special tokens

    Returns:
        BPETokenizer instance
    """
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]

    print(f"📁 Loading tokenizer from {vocab_path} and {merges_path}")

    # Load vocabulary
    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    # Convert: vocab_data has token_str -> token_id, we need token_id -> token_bytes
    vocab = {}
    for token_str, token_id in vocab_data.items():
        try:
            token_bytes = token_str.encode("utf-8")
        except UnicodeEncodeError:
            token_bytes = token_str.encode("latin-1")
        vocab[token_id] = token_bytes

    # Load merges
    merges = []
    with open(merges_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                if " " in line:
                    parts = line.split(" ")
                    if len(parts) >= 2:
                        mid_point = len(parts) // 2
                        token1 = " ".join(parts[:mid_point]).encode("utf-8")
                        token2 = " ".join(parts[mid_point:]).encode("utf-8")
                        merges.append((token1, token2))

    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    print(f"✅ Loaded tokenizer: {len(vocab)} vocab, {len(merges)} merges, special tokens: {special_tokens}")

    return tokenizer


def tokenize_file(
    tokenizer: BPETokenizer,
    input_path: str,
    output_path: Optional[str] = None,
    output_format: str = "ids",
    show_stats: bool = True,
) -> List[int]:
    """
    Tokenize an entire file.

    Args:
        tokenizer: BPE tokenizer instance
        input_path: Path to input text file
        output_path: Optional path to save tokenized output
        output_format: Output format - "ids", "text", or "json"
        show_stats: Whether to show tokenization statistics

    Returns:
        List of token IDs
    """
    print(f"🔤 Tokenizing file: {input_path}")

    # Use memory-efficient streaming tokenization
    start_time = time.time()
    
    print("   Using streaming tokenization to avoid memory issues...")
    
    token_count = 0
    token_ids = []  # Only collect if no output file or format requires it
    
    # Stream tokens directly to file for "ids" format to save memory
    if output_path and output_format == "ids":
        output_path = Path(output_path)
        with open(input_path, "r", encoding="utf-8") as input_f, \
             open(output_path, "w") as output_f:
            
            first_token = True
            for token_id in tokenizer.encode_iterable(input_f):
                if not first_token:
                    output_f.write(" ")
                output_f.write(str(token_id))
                first_token = False
                token_count += 1
                
                # Progress indicator for large files
                if token_count % 100000 == 0:
                    print(f"   • Processed {token_count:,} tokens...")
        
        print(f"💾 Saved token IDs to {output_path}")
        token_ids = []  # Don't keep in memory
    else:
        # For other formats or no output file, collect tokens
        with open(input_path, "r", encoding="utf-8") as f:
            token_ids = list(tokenizer.encode_iterable(f))
        token_count = len(token_ids)
    
    # Calculate text length for stats (if needed)
    if show_stats:
        with open(input_path, "r", encoding="utf-8") as f:
            text_len = sum(len(line) for line in f)
    else:
        text_len = 0

    end_time = time.time()
    tokenization_time = end_time - start_time

    if show_stats:
        print(f"📊 Tokenization stats:")
        print(f"   • Input characters: {text_len:,}")
        print(f"   • Output tokens: {token_count:,}")
        print(f"   • Compression ratio: {text_len / token_count:.2f}x")
        print(f"   • Time: {tokenization_time:.3f}s")
        print(f"   • Speed: {text_len / tokenization_time / 1000:.1f}K chars/sec")

    # Save output if path provided (only for non-ids formats, since ids was already streamed)
    if output_path and output_format != "ids":
        output_path = Path(output_path)

        if output_format == "json":
            # Save as JSON array
            with open(output_path, "w") as f:
                json.dump(token_ids, f)
            print(f"💾 Saved tokens as JSON to {output_path}")

        elif output_format == "text":
            # Save as human-readable text with token boundaries
            tokens_text = []
            for token_id in token_ids[:100]:  # Show first 100 tokens
                token_bytes = tokenizer.vocab[token_id]
                token_str = token_bytes.decode("utf-8", errors="replace")
                tokens_text.append(f"{token_id}:{repr(token_str)}")

            with open(output_path, "w", encoding="utf-8") as f:
                f.write("Token ID : Token String\n")
                f.write("=" * 50 + "\n")
                for token_text in tokens_text:
                    f.write(token_text + "\n")
                if len(token_ids) > 100:
                    f.write(f"\n... and {len(token_ids) - 100} more tokens")
            print(f"💾 Saved readable tokens to {output_path}")

    return token_ids


def detokenize_file(
    tokenizer: BPETokenizer, input_path: str, output_path: Optional[str] = None, input_format: str = "ids"
) -> str:
    """
    Detokenize token IDs back to text.

    Args:
        tokenizer: BPE tokenizer instance
        input_path: Path to tokenized input file
        output_path: Optional path to save detokenized text
        input_format: Input format - "ids" or "json"

    Returns:
        Detokenized text string
    """
    print(f"🔠 Detokenizing file: {input_path}")

    # Read token IDs
    if input_format == "ids":
        with open(input_path, "r") as f:
            content = f.read().strip()
            if not content:
                token_ids = []
            else:
                token_ids = [int(x) for x in content.split()]
    elif input_format == "json":
        with open(input_path, "r") as f:
            token_ids = json.load(f)
    else:
        raise ValueError(f"Unknown input format: {input_format}")

    start_time = time.time()

    # Detokenize
    text = tokenizer.decode(token_ids)

    end_time = time.time()
    detokenization_time = end_time - start_time

    print(f"📊 Detokenization stats:")
    print(f"   • Input tokens: {len(token_ids):,}")
    print(f"   • Output characters: {len(text):,}")
    print(f"   • Time: {detokenization_time:.3f}s")
    print(f"   • Speed: {len(token_ids) / detokenization_time / 1000:.1f}K tokens/sec")

    # Save output if path provided
    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"💾 Saved detokenized text to {output_path}")

    return text


def interactive_mode(tokenizer: BPETokenizer):
    """Interactive tokenization mode."""
    print("\n🤖 Interactive BPE Tokenization Mode")
    print("Commands:")
    print("  encode <text>     - Tokenize text")
    print("  decode <ids>      - Detokenize space-separated token IDs")
    print("  stats             - Show tokenizer statistics")
    print("  quit/exit         - Exit interactive mode")
    print("-" * 50)

    while True:
        try:
            command = input("\n>>> ").strip()
            if not command:
                continue

            if command.lower() in ["quit", "exit", "q"]:
                break

            parts = command.split(" ", 1)
            cmd = parts[0].lower()

            if cmd == "encode" and len(parts) > 1:
                text = parts[1]
                token_ids = tokenizer.encode(text)
                print(f'Text: "{text}"')
                print(f"Token IDs: {token_ids}")
                print(f"Length: {len(token_ids)} tokens")

                # Show token breakdown for short sequences
                if len(token_ids) <= 20:
                    print("Token breakdown:")
                    for i, token_id in enumerate(token_ids):
                        token_bytes = tokenizer.vocab[token_id]
                        token_str = token_bytes.decode("utf-8", errors="replace")
                        print(f"  {i}: {token_id} -> {repr(token_str)}")

            elif cmd == "decode" and len(parts) > 1:
                try:
                    token_ids = [int(x) for x in parts[1].split()]
                    text = tokenizer.decode(token_ids)
                    print(f"Token IDs: {token_ids}")
                    print(f'Text: "{text}"')
                except ValueError as e:
                    print(f"Error: Invalid token IDs - {e}")

            elif cmd == "stats":
                vocab_size = len(tokenizer.vocab)
                special_count = len(tokenizer.special_tokens)
                merge_count = len(tokenizer.merges)
                print(f"Tokenizer statistics:")
                print(f"  • Vocabulary size: {vocab_size:,}")
                print(f"  • Special tokens: {special_count}")
                print(f"  • Merge rules: {merge_count:,}")
                print(f"  • Special tokens: {tokenizer.special_tokens}")

            else:
                print("Unknown command. Type 'encode <text>', 'decode <ids>', 'stats', or 'quit'.")

        except KeyboardInterrupt:
            print("\n\nExiting interactive mode...")
            break
        except Exception as e:
            print(f"Error: {e}")


def parse_args():
    """Parse command-line arguments for BPE tokenization."""
    parser = argparse.ArgumentParser(
        description="Tokenize or detokenize text using a trained BPE tokenizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Tokenize a text file
  python code/bpe_tokenize.py \\
    --vocab tokenizer_output/vocab.json \\
    --merges tokenizer_output/merges.txt \\
    --input document.txt \\
    --output document.tokens

  # Detokenize token IDs back to text
  python code/bpe_tokenize.py \\
    --vocab tokenizer_output/vocab.json \\
    --merges tokenizer_output/merges.txt \\
    --input document.tokens \\
    --output document_reconstructed.txt \\
    --mode detokenize

  # Interactive mode for testing
  python code/bpe_tokenize.py \\
    --vocab tokenizer_output/vocab.json \\
    --merges tokenizer_output/merges.txt \\
    --interactive

  # Process multiple files with custom output format
  python code/bpe_tokenize.py \\
    --vocab my_tokenizer/vocab.json \\
    --merges my_tokenizer/merges.txt \\
    --input corpus.txt \\
    --output corpus.json \\
    --output-format json
        """,
    )

    # Required arguments
    parser.add_argument("--vocab", "-v", type=str, required=True, help="Path to vocab.json file")
    parser.add_argument("--merges", "-m", type=str, required=True, help="Path to merges.txt file")

    # Input/Output
    parser.add_argument("--input", "-i", type=str, help="Input file path")
    parser.add_argument("--output", "-o", type=str, help="Output file path")

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["tokenize", "detokenize"],
        default="tokenize",
        help="Operation mode (default: tokenize)",
    )

    # Format options
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["ids", "json", "text"],
        default="ids",
        help="Output format for tokenization (default: ids)",
    )
    parser.add_argument(
        "--input-format",
        type=str,
        choices=["ids", "json"],
        default="ids",
        help="Input format for detokenization (default: ids)",
    )

    # Special tokens
    parser.add_argument(
        "--special-tokens",
        "-s",
        type=str,
        nargs="*",
        default=["<|endoftext|>"],
        help="Special tokens list (default: ['<|endoftext|>'])",
    )

    # Interaction modes
    parser.add_argument("--interactive", action="store_true", help="Interactive mode for testing tokenization")
    parser.add_argument("--text", type=str, help="Direct text input for quick tokenization")

    # Options
    parser.add_argument("--no-stats", action="store_true", help="Disable statistics output")

    return parser.parse_args()


def main():
    """Main function for BPE tokenization."""
    args = parse_args()

    print("🔤 BPE Tokenizer")
    print("=" * 30)

    # Validate arguments
    if not os.path.exists(args.vocab):
        print(f"❌ Error: Vocab file '{args.vocab}' not found")
        return 1

    if not os.path.exists(args.merges):
        print(f"❌ Error: Merges file '{args.merges}' not found")
        return 1

    try:
        # Load tokenizer
        tokenizer = load_tokenizer(args.vocab, args.merges, args.special_tokens)

        # Interactive mode
        if args.interactive:
            interactive_mode(tokenizer)
            return 0

        # Direct text input
        if args.text:
            print(f'\n🔤 Tokenizing direct input: "{args.text}"')
            token_ids = tokenizer.encode(args.text)
            print(f"Token IDs: {token_ids}")
            print(f"Token count: {len(token_ids)}")

            # Show token breakdown
            print("Token breakdown:")
            for i, token_id in enumerate(token_ids):
                token_bytes = tokenizer.vocab[token_id]
                token_str = token_bytes.decode("utf-8", errors="replace")
                print(f"  {i}: {token_id} -> {repr(token_str)}")

            return 0

        # File processing mode
        if not args.input:
            print("❌ Error: No input specified. Use --input, --text, or --interactive")
            return 1

        if not os.path.exists(args.input):
            print(f"❌ Error: Input file '{args.input}' not found")
            return 1

        show_stats = not args.no_stats

        if args.mode == "tokenize":
            token_ids = tokenize_file(tokenizer, args.input, args.output, args.output_format, show_stats)
            print(f"✅ Tokenization completed: {len(token_ids)} tokens")

        elif args.mode == "detokenize":
            text = detokenize_file(tokenizer, args.input, args.output, args.input_format)
            print(f"✅ Detokenization completed: {len(text)} characters")

        return 0

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
