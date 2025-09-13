# CS336 Assignment 1 - Code Directory

Complete implementation of transformer language models from scratch, including BPE tokenization, training, and text generation.

## Quick Start

```bash
# 1. Train a BPE tokenizer
uv run python code/python/train_bpe_tokenizer.py --input data/corpus.txt --output tokenizer --vocab-size 32000

# 2. Train a transformer model
uv run python code/python/train.py --train_data data/train.npy --val_data data/val.npy --vocab_size 32000

# 3. Generate text
uv run python code/python/decode.py --checkpoint checkpoints/final_checkpoint.pt --prompt "Once upon a time"
```

---

## 📁 Scripts Overview

This directory contains five main Python scripts for the complete transformer training pipeline:

1. **`train_bpe_tokenizer.py`** - Train BPE tokenizers from text data
2. **`bpe_tokenize.py`** - Tokenize/detokenize text using trained BPE tokenizers
3. **`train.py`** - Train transformer language models
4. **`decode.py`** - Generate text from trained models
5. **`transformers.py`** - Core transformer implementations and utilities
6. **`bpe_tokenizer.py`** - BPE tokenizer implementation

---

## 🚀 BPE Tokenizer Training

### train_bpe_tokenizer.py

Train Byte Pair Encoding tokenizers for your text data.

#### Basic Usage
```bash
# Train with default settings
uv run python code/python/train_bpe_tokenizer.py \
    --input data/corpus.txt \
    --output my_tokenizer \
    --vocab-size 32000
```

#### Complete Example
```bash
# Train tokenizer for transformer models
uv run python code/python/train_bpe_tokenizer.py \
    --input data/tinystories_train.txt \
    --output tokenizer_output \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>" "<|pad|>" "<|unk|>" \
    --test-text "Once upon a time" \
    --verbose
```

#### Training Modes

**Full Training (default)**
```bash
# Complete training in one step
uv run python code/python/train_bpe_tokenizer.py \
    --input data/corpus.txt \
    --output my_tokenizer \
    --vocab-size 32000 \
    --mode full
```

**Extract Word Frequencies Only**
```bash
# Extract and save word frequencies without training
uv run python code/python/train_bpe_tokenizer.py \
    --input data/corpus.txt \
    --word-freqs-file word_freqs.json \
    --mode extract-freqs
```

**Train from Pre-extracted Frequencies**
```bash
# Train BPE from previously extracted word frequencies
uv run python code/python/train_bpe_tokenizer.py \
    --word-freqs-file word_freqs.json \
    --output my_tokenizer \
    --vocab-size 32000 \
    --mode train-from-freqs
```

#### Arguments

**Required (mode-dependent):**
- `--input, -i`: Path to training text file (required for `full` and `extract-freqs` modes)
- `--output, -o`: Output directory for tokenizer files (required for `full` and `train-from-freqs` modes)
- `--vocab-size, -v`: Target vocabulary size (required for `full` and `train-from-freqs` modes)
- `--word-freqs-file`: Path to word frequencies JSON file (required for `extract-freqs` and `train-from-freqs` modes)

**Optional:**
- `--mode`: Training mode - `full`, `extract-freqs`, or `train-from-freqs` (default: `full`)
- `--special-tokens, -s`: Special tokens (default: `["<|endoftext|>"]`)
- `--test-text`: Sample text to test after training
- `--verbose`: Enable verbose training output

#### Output Files
The script creates these files in the output directory:
- `vocab.json` - Vocabulary mapping
- `merges.txt` - Merge rules 
- `training_stats.txt` - Training statistics

---

## 🔤 Text Tokenization

### bpe_tokenize.py

Tokenize and detokenize text using trained BPE tokenizers.

#### Usage Modes

**1. File Tokenization**
```bash
# Tokenize file to token IDs
uv run python code/python/bpe_tokenize.py \
    --vocab tokenizer_output/vocab.json \
    --merges tokenizer_output/merges.txt \
    --input document.txt \
    --output document.tokens

# Tokenize to JSON format
uv run python code/python/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --input corpus.txt \
    --output corpus.json \
    --output-format json
```

**2. File Detokenization**
```bash
# Convert token IDs back to text
uv run python code/python/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --input document.tokens \
    --output document_reconstructed.txt \
    --mode detokenize
```

**3. Direct Text Processing**
```bash
# Tokenize text directly
uv run python code/python/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --text "Hello world! This is a test."
```

**4. Interactive Mode**
```bash
# Interactive tokenization session
uv run python code/python/bpe_tokenize.py \
    --vocab my_tokenizer/vocab.json \
    --merges my_tokenizer/merges.txt \
    --interactive
```

#### Arguments

**Required:**
- `--vocab, -v`: Path to vocab.json file
- `--merges, -m`: Path to merges.txt file

**Input/Output:**
- `--input, -i`: Input file path
- `--output, -o`: Output file path
- `--text`: Direct text input

**Modes:**
- `--mode`: `tokenize` or `detokenize` (default: `tokenize`)
- `--interactive`: Interactive mode

**Formats:**
- `--output-format`: `ids`, `json`, or `text` (default: `ids`)
- `--input-format`: `ids` or `json` (default: `ids`)

---

## 🧠 Model Training

### train.py

Train transformer language models with comprehensive hyperparameter control.

#### Quick Start
```bash
# Basic training run
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --d_model 512 \
    --num_heads 8 \
    --num_layers 6 \
    --max_steps 100000
```

#### Key Features

**Memory-Efficient Data Loading**
- Uses `np.memmap` to handle datasets larger than available RAM
- Automatic detection of dataset size and efficient batch sampling
- Support for both `.npy` files and memory-mapped arrays

**Model Architecture Control**
- `--d_model`: Hidden dimension (default: 512)
- `--num_heads`: Number of attention heads (default: 8) 
- `--d_ff`: Feed-forward dimension (default: 2048)
- `--num_layers`: Number of transformer layers (default: 6)
- `--vocab_size`: Vocabulary size (default: 32000)
- `--context_length`: Maximum sequence length (default: 1024)
- `--rope_theta`: RoPE theta parameter (default: 10000.0)

**Training Configuration**
- `--batch_size`: Batch size (default: 32)
- `--learning_rate`: Peak learning rate (default: 3e-4)
- `--min_learning_rate`: Minimum learning rate (default: 3e-5)
- `--warmup_steps`: Learning rate warmup steps (default: 1000)
- `--max_steps`: Maximum training steps (default: 100000)
- `--grad_clip`: Gradient clipping max norm (default: 1.0)

#### Advanced Features

**Learning Rate Scheduling:**
- Cosine annealing with linear warmup
- Configurable warmup period and decay schedule

**Gradient Optimization:**
- AdamW optimizer with configurable betas and weight decay
- Global gradient norm clipping across all parameters
- Automatic mixed precision support (float16/float32)

**Checkpointing:**
- Automatic checkpoint saving at configurable intervals
- Resume training from any checkpoint
- Saves model state, optimizer state, and training step

#### Example Configurations

**Small Model (Testing)**
```bash
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --d_model 128 \
    --num_heads 4 \
    --d_ff 512 \
    --num_layers 2 \
    --batch_size 16 \
    --max_steps 1000 \
    --device cpu
```

**Large Model (GTX 1080 Ti)**
```bash
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --d_model 768 \
    --num_heads 12 \
    --d_ff 3072 \
    --num_layers 12 \
    --batch_size 8 \
    --dtype float16 \
    --device cuda
```

**Resume Training**
```bash
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --resume_from checkpoints/checkpoint_step_50000.pt \
    --max_steps 150000
```

**Weights & Biases Logging**
```bash
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --use_wandb \
    --wandb_project my-transformer \
    --wandb_run_name experiment-1
```

#### Output Structure
```
checkpoints/
├── config.json              # Training configuration
├── checkpoint_step_5000.pt  # Periodic checkpoints
├── checkpoint_step_10000.pt
├── ...
└── final_checkpoint.pt      # Final model state
```

---

## 🎯 Text Generation

### decode.py

Generate text from trained transformer models with various sampling strategies.

#### Basic Usage

**Single Text Generation**
```bash
# Basic generation with minimal arguments
uv run python code/python/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --prompt "Once upon a time" \
    --max-tokens 50

# With custom sampling parameters  
uv run python code/python/decode.py \
    --checkpoint checkpoints/checkpoint_step_10000.pt \
    --prompt "The future of AI is" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9 \
    --device cuda
```

**Interactive Mode**
```bash
# Start interactive session for multiple prompts
uv run python code/python/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --interactive \
    --temperature 0.7 \
    --max-tokens 75
```

**Reproducible Generation**
```bash
# Set seed for consistent results
uv run python code/python/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --prompt "Hello world" \
    --seed 42 \
    --temperature 1.0
```

#### Auto-Detection Features

**Automatic Configuration Loading**
The script automatically searches for `config.json` in:
1. Same directory as the checkpoint file
2. Parent directory of the checkpoint file
3. Specified via `--config` argument

**Automatic Tokenizer Detection**
The script searches for `vocab.json` and `merges.txt` in:
1. Same directory as the checkpoint
2. Parent directory of the checkpoint  
3. `tokenizer_output/` directory
4. Current directory
5. Specified via `--vocab` and `--merges` arguments

#### Arguments

**Required:**
- `--checkpoint`: Path to model checkpoint (.pt file)

**Generation Parameters:**
- `--prompt`: Input text prompt (required unless using `--interactive`)
- `--max-tokens`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature - higher = more creative (default: 1.0)
- `--top-p`: Top-p (nucleus) sampling threshold (optional)
- `--seed`: Random seed for reproducible generation (optional)

**File Paths (Auto-detected if not specified):**
- `--config`: Path to config.json file
- `--vocab`: Path to vocab.json file
- `--merges`: Path to merges.txt file

**Other Options:**
- `--device`: Device to use - "cpu", "cuda", etc. (auto-detected)
- `--special-tokens`: Special tokens for tokenizer (default: ["<|endoftext|>"])
- `--interactive`: Interactive mode for multiple prompts

---

## 🔄 Complete Workflow

Here's the complete pipeline from raw text to trained model and text generation:

### 1. Prepare Your Data
```bash
# Download or prepare your text corpus
# Example: TinyStories dataset
mkdir -p data
cd data
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
cd ..
```

### 2. Train BPE Tokenizer
```bash
uv run python code/python/train_bpe_tokenizer.py \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output tokenizer_tinystories \
    --vocab-size 32000 \
    --special-tokens "<|endoftext|>"
```

### 3. Tokenize Training Data
```bash
# Tokenize training data
uv run python code/python/bpe_tokenize.py \
    --vocab tokenizer_tinystories/vocab.json \
    --merges tokenizer_tinystories/merges.txt \
    --input data/TinyStoriesV2-GPT4-train.txt \
    --output data/train.tokens \
    --output-format json

# Tokenize validation data
uv run python code/python/bpe_tokenize.py \
    --vocab tokenizer_tinystories/vocab.json \
    --merges tokenizer_tinystories/merges.txt \
    --input data/TinyStoriesV2-GPT4-valid.txt \
    --output data/val.tokens \
    --output-format json
```

### 4. Convert to NumPy Arrays
```python
import numpy as np
import json

# Load and convert training data
with open('data/train.tokens', 'r') as f:
    train_tokens = json.load(f)
np.save('data/train.npy', np.array(train_tokens, dtype=np.int32))

# Load and convert validation data
with open('data/val.tokens', 'r') as f:
    val_tokens = json.load(f)
np.save('data/val.npy', np.array(val_tokens, dtype=np.int32))
```

### 5. Train Transformer Model
```bash
uv run python code/python/train.py \
    --train_data data/train.npy \
    --val_data data/val.npy \
    --d_model 768 \
    --num_heads 12 \
    --d_ff 3072 \
    --num_layers 12 \
    --vocab_size 32000 \
    --batch_size 8 \
    --max_steps 50000 \
    --checkpoint_dir checkpoints \
    --device cuda
```

### 6. Generate Text
```bash
# The decode script auto-detects tokenizer files
uv run python code/python/decode.py \
    --checkpoint checkpoints/final_checkpoint.pt \
    --prompt "Once upon a time, there was a little girl" \
    --max-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9
```

---

## 💾 Memory Optimization for GTX 1080 Ti (11GB)

### Recommended Model Configurations

**Conservative (Safe) - ~6-7GB**
```bash
uv run python code/python/train.py \
    --d_model 768 --num_heads 12 --d_ff 3072 --num_layers 12 \
    --batch_size 8 --dtype float16
```

**Optimized (Balanced) - ~8-9GB**
```bash
uv run python code/python/train.py \
    --d_model 1024 --num_heads 16 --d_ff 4096 --num_layers 20 \
    --batch_size 4 --dtype float16
```

**Maximum (Aggressive) - ~10-10.5GB**
```bash
uv run python code/python/train.py \
    --d_model 1280 --num_heads 20 --d_ff 5120 --num_layers 30 \
    --batch_size 2 --dtype float16
```

### Memory Optimization Tips
- Use `--dtype float16` to halve memory usage
- Reduce `--batch_size` if out of memory
- Reduce `--context_length` (512 instead of 1024)
- Monitor with `nvidia-smi -l 1`

---

## 🛠️ Troubleshooting

### Common Issues

**BPE Tokenizer Training**
- **"Input file not found"**: Check file path and permissions
- **"Vocab size too small"**: Must be > 256 (base byte vocabulary)
- **"Unicode decode error"**: The script handles this automatically

**Model Training**
- **Out of Memory**: Reduce batch size, use float16, reduce model size
- **Slow Training**: Enable `--compile`, increase batch size if memory allows
- **Convergence Issues**: Adjust learning rate, increase warmup steps

**Text Generation**
- **"Config file not found"**: Ensure `config.json` exists in checkpoint directory
- **"Could not find tokenizer"**: Ensure `vocab.json` and `merges.txt` exist
- **"CUDA out of memory"**: Use `--device cpu` or reduce model size

### Performance Monitoring

**GPU Usage**
```bash
# Monitor GPU usage during training
nvidia-smi -l 1

# Check GPU memory with Python
python -c "import torch; print(f'GPU memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB')"
```

**Training Speed**
- Check tokens/second in training logs
- Higher batch sizes generally improve throughput
- Mixed precision (`--dtype float16`) speeds up training

---

## 📊 Integration with Other Tools

### Programmatic Usage

**Import and use functions directly:**
```python
from code.python.decode import load_model_from_checkpoint, load_tokenizer_from_files, decode_with_tokenizer
from code.python.train import create_model_and_optimizer
from code.python.bpe_tokenizer import BPETokenizer, train_bpe

# Load components
model = load_model_from_checkpoint("checkpoints/model.pt")
tokenizer = load_tokenizer_from_files("vocab.json", "merges.txt")

# Generate text
text = decode_with_tokenizer(model, tokenizer, "Hello world", max_tokens=50)
```

### Weights & Biases Integration
```bash
# Enable logging during training
uv run python code/python/train.py \
    --use_wandb \
    --wandb_project transformer-experiments \
    --wandb_run_name tinystories-768d-12l
```

---

## ✅ Testing Your Setup

### Quick Test Pipeline
```bash
# 1. Create test data
echo "Hello world! This is a test. The quick brown fox jumps over the lazy dog." > test.txt

# 2. Train small tokenizer
uv run python code/python/train_bpe_tokenizer.py --input test.txt --output test_tokenizer --vocab-size 500

# 3. Test tokenization
uv run python code/python/bpe_tokenize.py --vocab test_tokenizer/vocab.json --merges test_tokenizer/merges.txt --text "Hello world!"

# 4. Clean up
rm test.txt && rm -rf test_tokenizer
```

This completes the documentation for all scripts in the code directory. Each script provides comprehensive help via `--help` flag for detailed argument information.