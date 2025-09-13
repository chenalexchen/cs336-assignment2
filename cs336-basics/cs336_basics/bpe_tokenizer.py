import os
import json
from pathlib import Path
from collections import defaultdict
import regex as re
from multiprocessing import Pool, cpu_count


class TokenNode:
    def __init__(self, token_id: int):
        self.token_id = token_id
        self.next = None
        self.prev = None


def find_chunk_boundaries(file, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def process_chunk(args):
    """Process a single chunk of text for pre-tokenization."""
    chunk_text, special_tokens = args

    # Pre-compile the pattern for better performance
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    pat = re.compile(PAT)

    # Split text on special tokens BEFORE pre-tokenization
    if special_tokens:
        # Escape special tokens for regex and join with |
        escaped_tokens = [re.escape(token) for token in special_tokens]
        split_pattern = "|".join(escaped_tokens)
        text_chunks = re.split(split_pattern, chunk_text)
    else:
        text_chunks = [chunk_text]

    # Extract pre-tokens and count their frequencies
    word_freqs = {}

    for chunk in text_chunks:
        if chunk:  # Skip empty chunks
            # Pre-tokenize each chunk separately using faster iteration
            matches = pat.findall(chunk)
            for word_text in matches:
                if word_text:  # Skip empty words
                    word_bytes = tuple(word_text.encode("utf-8"))  # Use tuple as key
                    word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + 1

    return word_freqs


def extract_word_freqs(input_path: str | os.PathLike, special_tokens: list[str]) -> dict[tuple[int, ...], int]:
    """Extracts word frequencies from a text file."""
    # Use parallel processing for pre-tokenization
    # Use most cores but leave 1-2 for system responsiveness
    num_processes = max(1, cpu_count() - 1)

    # For small files, don't bother with parallelization
    file_size = os.path.getsize(input_path)
    if file_size < 1024 * 1024:  # Less than 1MB
        # Fall back to sequential processing
        with open(input_path, encoding="utf-8") as f:
            text = f.read()

        chunk_results = [process_chunk((text, special_tokens))]
    else:
        # Use parallel processing with chunking
        with open(input_path, "rb") as f:
            # Use the first special token for chunking, or <|endoftext|> as default
            split_token = special_tokens[0] if special_tokens else "<|endoftext|>"
            split_token_bytes = split_token.encode("utf-8")

            boundaries = find_chunk_boundaries(f, num_processes, split_token_bytes)

            # Read chunks
            chunks = []
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                chunk_bytes = f.read(end - start)
                chunk_text = chunk_bytes.decode("utf-8", errors="ignore")
                chunks.append((chunk_text, special_tokens))

        # Process chunks in parallel
        with Pool(num_processes) as pool:
            chunk_results = pool.map(process_chunk, chunks)

    # Combine results from all chunks more efficiently
    word_freqs = {}
    for chunk_result in chunk_results:
        for word_bytes, freq in chunk_result.items():
            word_freqs[word_bytes] = word_freqs.get(word_bytes, 0) + freq

    import json

    with open("output/tokenizer_inter_op/python/word_freqs.json", "w") as f:
        string_keyed_word_freqs = {",".join(map(str, k)): v for k, v in word_freqs.items()}
        json.dump(string_keyed_word_freqs, f)

    return word_freqs


def train_bpe_merges(word_freqs: dict, vocab: dict, target_merges: int) -> list[tuple[bytes, bytes]]:
    """
    Shared merge logic for BPE training.

    Args:
        word_freqs: Dictionary mapping word tuples to frequencies
        vocab: Current vocabulary mapping token IDs to bytes
        target_merges: Number of merges to perform

    Returns:
        List of merge rules as (token1_bytes, token2_bytes) tuples
    """
    merges = []
    next_token_id = max(vocab.keys()) + 1

    # Initialize pair counts once
    pair_counts = defaultdict(int)
    for word_bytes, freq in word_freqs.items():
        word_len = len(word_bytes)
        for i in range(word_len - 1):
            pair = (word_bytes[i], word_bytes[i + 1])
            pair_counts[pair] += freq

    for _ in range(target_merges):
        # Find most frequent pair with deterministic tie-breaking
        if not pair_counts:
            break

        # Get the best pair directly - optimize by avoiding repeated vocab lookups
        best_pair = None
        best_count = 0
        best_tie_breaker = None

        for pair, count in pair_counts.items():
            tie_breaker = (vocab[pair[0]], vocab[pair[1]])
            if count > best_count or (count == best_count and tie_breaker > best_tie_breaker):
                best_pair = pair
                best_count = count
                best_tie_breaker = tie_breaker

        most_frequent_pair = best_pair

        max_count = pair_counts[most_frequent_pair]
        if max_count == 0:
            break

        token1_id, token2_id = most_frequent_pair

        # Create new token
        new_token_id = next_token_id
        next_token_id += 1

        # Merge the tokens in vocabulary
        token1_bytes = vocab[token1_id]
        token2_bytes = vocab[token2_id]
        vocab[new_token_id] = token1_bytes + token2_bytes
        merges.append((token1_bytes, token2_bytes))

        # Incrementally update pair counts and word frequencies
        new_word_freqs = defaultdict(int)

        for word_bytes, freq in word_freqs.items():
            # Quick check: does this word even contain the tokens we're merging?
            if token1_id not in word_bytes or token2_id not in word_bytes:
                # No change needed, just copy over
                new_word_freqs[word_bytes] += freq
                continue

            word = list(word_bytes)
            word_len = len(word)

            # Check if this word contains the consecutive pair to merge
            has_consecutive_pair = False
            for i in range(word_len - 1):
                if word[i] == token1_id and word[i + 1] == token2_id:
                    has_consecutive_pair = True
                    break

            if has_consecutive_pair:
                # Remove old pair counts for this word
                for i in range(word_len - 1):
                    old_pair = (word[i], word[i + 1])
                    pair_counts[old_pair] -= freq
                    if pair_counts[old_pair] <= 0:
                        del pair_counts[old_pair]

                # Apply merge efficiently
                i = 0
                word_len_minus_1 = word_len - 1
                while i < word_len_minus_1:
                    if word[i] == token1_id and word[i + 1] == token2_id:
                        word[i] = new_token_id
                        word.pop(i + 1)
                        word_len_minus_1 -= 1  # Adjust length after pop
                    else:
                        i += 1

                # Add new pair counts for the updated word
                new_word_len = len(word)
                for i in range(new_word_len - 1):
                    new_pair = (word[i], word[i + 1])
                    pair_counts[new_pair] += freq

                # Add updated word to new frequencies
                new_word_key = tuple(word)
                new_word_freqs[new_word_key] += freq
            else:
                # Word contains the tokens but not consecutively, no change needed
                new_word_freqs[word_bytes] += freq

        word_freqs = dict(new_word_freqs)

    return merges


def train_bpe_from_freqs(
    word_freqs: dict, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Train BPE from pre-computed word frequencies.

    Args:
        word_freqs: Dictionary mapping word tuples to frequencies
        vocab_size: Target vocabulary size
        special_tokens: List of special tokens

    Returns:
        vocab: Vocabulary mapping token IDs to bytes
        merges: List of merge rules
    """
    # Initialize vocabulary with base bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocab
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    if not word_freqs:
        return vocab, []

    # Calculate target number of merges
    target_merges = vocab_size - len(vocab)
    if target_merges <= 0:
        return vocab, []

    # Use shared merge logic
    merges = train_bpe_merges(word_freqs, vocab, target_merges)

    return vocab, merges


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


def train_bpe(
    input_path: str | os.PathLike, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """
    Trains a (byte-level) BPE tokenizer with proper pre-tokenization.

    Args:
        input_path: Path to a text file with BPE tokenizer training data.
        vocab_size: A positive integer that defines the maximum final vocabulary size (including the
                    initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
        special_tokens: A list of strings to add to the vocabulary. These special tokens do not
                        otherwise affect BPE training.
    Returns:
        vocab: The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
        merges: A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
    """
    # Initialize vocabulary with base bytes (0-255)
    vocab = {i: bytes([i]) for i in range(256)}
    next_token_id = 256

    # Add special tokens to vocab
    for special_token in special_tokens:
        vocab[next_token_id] = special_token.encode("utf-8")
        next_token_id += 1

    word_freqs = extract_word_freqs(input_path, special_tokens)

    if not word_freqs:
        return vocab, []

    # Calculate target number of merges
    target_merges = vocab_size - len(vocab)
    if target_merges <= 0:
        return vocab, []

    # Use shared merge logic
    merges = train_bpe_merges(word_freqs, vocab, target_merges)

    return vocab, merges


def decode(token_ids: list[int], vocab: dict[int, bytes]) -> str:
    """
    Decodes token IDs back to text.

    Args:
        token_ids: List of token IDs to decode
        vocab: Vocabulary mapping

    Returns:
        Decoded text string
    """
    byte_sequence = b"".join(vocab[token_id] for token_id in token_ids)
    return byte_sequence.decode("utf-8", errors="replace")


class BPETokenizer:
    """
    A Byte Pair Encoding (BPE) tokenizer.
    """

    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        """
        Initialize the BPE tokenizer.

        Args:
            vocab: Vocabulary mapping from token IDs to bytes
            merges: List of merge rules
            special_tokens: List of special tokens
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # Create reverse mapping
        self.bytes_to_id = {v: k for k, v in vocab.items()}

        # Pre-compile regex pattern for efficiency
        self.pat = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

        # Pre-compute merge priority lookup for efficiency
        self.merge_priorities = {}
        for i, (token1, token2) in enumerate(merges):
            merged_token = token1 + token2
            if merged_token in self.bytes_to_id:
                self.merge_priorities[(token1, token2)] = i

        # Create special token mappings
        self.special_token_to_id = {}
        for token in self.special_tokens:
            token_bytes = token.encode("utf-8")
            if token_bytes in self.bytes_to_id:
                self.special_token_to_id[token] = self.bytes_to_id[token_bytes]

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        Args:
            text: Text to encode

        Returns:
            List of token IDs
        """
        return self._encode_optimized(text)

    def decode(self, token_ids: list[int]) -> str:
        """
        Decode token IDs to text.

        Args:
            token_ids: List of token IDs

        Returns:
            Decoded text string
        """
        return decode(token_ids, self.vocab)

    def encode_iterable(self, text_iterable) -> list[int]:
        """
        Memory-efficient encoding of an iterable of text (e.g., file lines).

        Args:
            text_iterable: Iterable that yields text strings (e.g., open file)

        Yields:
            Token IDs one by one
        """
        # Read text in chunks to be memory efficient
        buffer = ""

        for line in text_iterable:
            buffer += line

            # Process complete lines or when buffer gets large
            if "\n" in buffer or len(buffer) > 1024:
                # Split on newlines and keep the last incomplete line in buffer
                lines = buffer.split("\n")
                buffer = lines[-1]  # Keep incomplete last line

                # Process complete lines (including empty lines)
                for complete_line in lines[:-1]:
                    # Always add newline back since split removed it
                    line_with_newline = complete_line + "\n"
                    token_ids = self.encode(line_with_newline)
                    for token_id in token_ids:
                        yield token_id

        # Process any remaining text in buffer
        if buffer:
            token_ids = self.encode(buffer)
            for token_id in token_ids:
                yield token_id

    def _encode_optimized(self, text: str) -> list[int]:
        """
        Optimized encoding that reuses precompiled patterns and lookups.
        """
        # Handle special tokens first
        result = []

        if self.special_tokens:
            # Sort by length (descending) to handle overlapping tokens correctly
            sorted_special_tokens = sorted(self.special_tokens, key=len, reverse=True)
            escaped_tokens = [re.escape(token) for token in sorted_special_tokens]
            special_pattern = "|".join(escaped_tokens)

            # Split text while preserving special tokens
            parts = re.split(f"({special_pattern})", text)

            for part in parts:
                if part in self.special_tokens:
                    # Add special token ID
                    special_token_bytes = part.encode("utf-8")
                    if special_token_bytes in self.bytes_to_id:
                        result.append(self.bytes_to_id[special_token_bytes])
                elif part:  # Non-empty regular text
                    # Encode using BPE
                    result.extend(self._encode_text_part_optimized(part))
        else:
            result.extend(self._encode_text_part_optimized(text))

        return result

    def _encode_text_part_optimized(self, text: str) -> list[int]:
        """
        Encode a text part using BPE with precompiled patterns and lookups.
        """
        result = []

        # Process each pre-token separately using precompiled pattern
        for match in self.pat.finditer(text):
            pre_token = match.group()
            if not pre_token:
                continue

            # Start with byte-level tokens for this pre-token
            tokens = [bytes([b]) for b in pre_token.encode("utf-8")]

            # Apply merges using priority-based approach with precomputed priorities
            while len(tokens) > 1:
                # Find the best merge available in current token sequence
                best_merge = None
                best_priority = len(self.merges)  # Higher than any real priority
                best_pos = -1

                for i in range(len(tokens) - 1):
                    pair = (tokens[i], tokens[i + 1])
                    if pair in self.merge_priorities:
                        priority = self.merge_priorities[pair]
                        if priority < best_priority:
                            best_merge = pair
                            best_priority = priority
                            best_pos = i

                if best_merge is None:
                    break

                # Apply the best merge
                token1, token2 = best_merge
                merged_token = token1 + token2
                new_tokens = tokens[:best_pos] + [merged_token] + tokens[best_pos + 2 :]
                tokens = new_tokens

            # Convert to token IDs and add to result
            for token in tokens:
                if token in self.bytes_to_id:
                    result.append(self.bytes_to_id[token])

        return result
