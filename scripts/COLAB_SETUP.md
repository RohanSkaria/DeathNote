# Google Colab Setup Guide

## Quick Start

### 1. Install Required Packages

Run this cell first in Colab:

```python
# Install PEFT for LoRA (optional but recommended)
!pip install peft -q

# Verify installation
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

### 2. Upload Your Data

**Option A: Upload directly to Colab**
```python
from google.colab import files
uploaded = files.upload()  # Upload protein_data_batch_1.csv
```

**Option B: Use Google Drive (Recommended)**
```python
from google.colab import drive
drive.mount('/content/drive')

# Your file should be at:
# /content/drive/MyDrive/Death Note/protein_data_batch_1.csv
```

**Option C: Download from URL**
```python
!wget -O /content/data/protein_data_batch_1.csv "YOUR_DOWNLOAD_URL"
```

### 3. Run Training Script

```python
# Copy your training script to Colab or upload it
# Then run:
exec(open('train_protein_classifier.py').read())
```

Or use `%run` magic command:
```python
%run train_protein_classifier.py
```

## Colab-Specific Configuration

### GPU Selection

**Free Tier (T4 GPU)**:
- Use 35M model: `USE_650M_MODEL = False`
- Batch size: 64-128
- May run out of memory with 650M

**Colab Pro (A100 GPU)**:
- Use 650M model: `USE_650M_MODEL = True`
- Batch size: 64
- LoRA recommended: `USE_LORA = True`

### Memory Management

If you get "Out of Memory" errors:

```python
# Reduce batch size
BATCH_SIZE = 32  # or even 16

# Increase gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 4

# Enable mixed precision (already enabled by default)
USE_MIXED_PRECISION = True

# Clear cache
import gc
torch.cuda.empty_cache()
gc.collect()
```

## File Paths in Colab

The script automatically detects files in these locations:
1. `/content/data/*.csv` (uploaded files)
2. `/content/drive/MyDrive/Death Note/protein_data_batch_1.csv` (Google Drive)
3. Current directory `*.csv` (fallback)

Models are saved to:
- `/content/drive/MyDrive/Death Note/models/` (persists after session)

## Troubleshooting

### "CUDA out of memory"
```python
# Solution 1: Reduce batch size
BATCH_SIZE = 32

# Solution 2: Use smaller model
USE_650M_MODEL = False

# Solution 3: Clear GPU cache
torch.cuda.empty_cache()
```

### "PEFT not found"
```python
!pip install peft
# Then restart runtime or re-import
```

### "File not found"
```python
# Check if file exists
import os
print(os.path.exists('/content/drive/MyDrive/Death Note/protein_data_batch_1.csv'))

# List files in directory
!ls -lh /content/drive/MyDrive/Death\ Note/
```

### Slow Training
- Check GPU is enabled: Runtime → Change runtime type → GPU
- Verify GPU usage: `!nvidia-smi`
- Use mixed precision: `USE_MIXED_PRECISION = True`

## Recommended Colab Workflow

1. **Mount Google Drive** (persistent storage)
```python
from google.colab import drive
drive.mount('/content/drive')
```

2. **Install packages**
```python
!pip install peft transformers torch pandas numpy scikit-learn tqdm -q
```

3. **Upload/Download data**
```python
# Option: Download from your source
# Or upload via Colab file browser
```

4. **Run training**
```python
# Load and run script
exec(open('train_protein_classifier.py').read())
```

5. **Download results**
```python
# Models are saved to Google Drive automatically
# Or download manually:
from google.colab import files
files.download('/content/drive/MyDrive/Death Note/models/best_model_state.bin')
```

## Performance Tips

### For T4 GPU (Free Tier)
- Use 35M model
- Batch size: 64-128
- Gradient accumulation: 2-4
- Expect ~2-3 hours for 3 epochs

### For A100 GPU (Pro)
- Use 650M model with LoRA
- Batch size: 64
- Gradient accumulation: 1-2
- Expect ~1-2 hours for 3 epochs

## Saving Progress

Models are automatically saved to Google Drive, so they persist after the session ends.

To resume training:
```python
# Load checkpoint
checkpoint = torch.load('/content/drive/MyDrive/Death Note/models/best_model_state.bin')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# Continue training...
```

