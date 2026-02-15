# CUDA/GPU Setup Guide

## Current Status

✓ **Hardware**: GRID P40-1Q GPU with CUDA 12.2 available
✗ **PyTorch**: Currently installed as CPU-only (2.10.0+cpu)
✗ **Python**: 3.14.2 (too new - PyTorch CUDA wheels only support up to Python 3.13)

## Why GPU is not working

Python 3.14 was just released and PyTorch doesn't have CUDA-enabled wheels for it yet. Only the CPU version is available on PyPI.

## How to Enable GPU

### Option 1: Downgrade Python (Recommended)

1. Create a new Python 3.13 virtual environment:
```powershell
python3.13 -m venv .venv_cu122
.venv_cu122\Scripts\activate
```

2. Install requirements:
```powershell
pip install -r requirements.txt
```

3. Reinstall PyTorch with CUDA 12.2 support:
```powershell
pip uninstall torch -y
pip install torch --index-url https://download.pytorch.org/whl/cu122
```

4. Verify GPU is working:
```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

### Option 2: Wait for PyTorch Python 3.14 Support

Monitor the [PyTorch website](https://pytorch.org/get-started/locally/) for Python 3.14 wheels, or check periodically for updates.

## Expected Performance Improvement

- **CPU Training**: ~21 seconds per batch → ~3 hours total for 3 epochs
- **GPU Training**: ~0.5-1 second per batch → ~5-10 minutes total for 3 epochs

That's a **20-36x speedup**!

## Notebook Changes Made

The notebook has been optimized to work on CPU:

1. **Max sequence length reduced** from 512 → 256 for faster processing
2. **Improved progress tracking** with real-time loss display
3. **Better status messages** indicating CPU mode with training time estimates
4. **GPU detection** with helpful instructions if GPU is available

These changes will make CPU training faster, but GPU is still highly recommended for serious training work.

## After Enabling GPU

Simply run the notebook as-is - it will automatically detect CUDA and use GPU for all computations.
