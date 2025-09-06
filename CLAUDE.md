# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is CS336 Spring 2025 Assignment 2 - Systems, focused on optimized Transformer language models and distributed training. The codebase consists of two main Python modules:

- `cs336-basics`: Points to `../cs336-assignment1` which contains the assignment 1 implementation with basic Transformer components, data handling, and optimization utilities
- `cs336_systems`: The main development module where optimized implementations, distributed training, and system optimizations are implemented

## Development Commands

### Environment Setup
```bash
# All commands use uv for dependency management
uv run python  # Launch Python with all dependencies installed
```

### Testing
```bash
# Run all tests
uv run pytest -v ./tests

# Run specific test modules
uv run pytest tests/test_attention.py
uv run pytest tests/test_ddp.py
uv run pytest tests/test_sharded_optimizer.py

# Run tests with XML output (for submission)
uv run pytest -v ./tests --junitxml=test_results.xml
```

### Submission
```bash
# Run tests and create submission package
./test_and_make_submission.sh
```

### Code Quality
```bash
# Linting (configured in pyproject.toml)
# Use ruff for linting - line length 120, some ignores for __init__.py files
```

## Code Architecture

### Core Components

**cs336-basics Module** (from `../cs336-assignment1/cs336_basics/`):
- `transformers.py`: Transformer architecture with custom Linear, Embedding, RMSNorm, MLP, Attention, and Transformer classes
- `train.py`: Training utilities and main training loop
- `bpe_tokenizer.py`: BPE tokenizer implementation
- `generate_text.py`: Text generation utilities
- `decode.py`: Decoding and inference functions

**cs336_systems Module** (`cs336_systems/`):
- Main development area for assignment 2 implementations
- Currently minimal - students implement optimized versions here

### Key Patterns

1. **Type Annotations**: Heavy use of jaxtyping for tensor shape annotations (e.g., `Float[Tensor, " ... d_in"]`)
2. **Custom Initialization**: Truncated normal initialization with fan-in/fan-out scaling for Linear layers
3. **Einstein Operations**: Extensive use of einops for tensor operations (`einsum`, `rearrange`)
4. **Distributed Training**: Test framework for DDP (Distributed Data Parallel) implementations
5. **FlashAttention Integration**: Test adapters for FlashAttention variants (PyTorch and Triton)

### Test Structure

Tests are organized around system components:
- `test_attention.py`: FlashAttention implementations and correctness
- `test_ddp.py`: Distributed Data Parallel training with bucketing strategies
- `test_ddp_individual_parameters.py`: Parameter-level DDP optimizations
- `test_sharded_optimizer.py`: Optimizer state sharding across devices
- `adapters.py`: Interface adapters for different attention implementations

### Dependencies

Key dependencies include:
- PyTorch 2.6.0+ (2.2.2 for Intel Macs)
- einops/einx for tensor operations
- pytest for testing
- wandb for experiment tracking
- jaxtyping for type annotations
- triton (implied by FlashAttention tests)

## Development Guidelines

- Use `uv run` prefix for all Python commands to ensure proper dependency resolution
- Follow existing type annotation patterns with jaxtyping
- When implementing optimizations in `cs336_systems`, you can copy code from the `../cs336-assignment1` implementation as a starting point
- The codebase expects implementations to be functionally equivalent to reference implementations
- Test files contain comprehensive correctness checks against reference implementations
- Always write new python code to cs336_systems