# Training Script V2 - Critical Upgrades

## Overview
This document describes the major upgrades made to `train_protein_classifier.py` to address statistical flaws and improve model performance.

## 🚀 Key Changes

### 1. **Model Upgrade: 35M → 650M with LoRA**
**Problem**: 35M model is too shallow for complex protein patterns
**Solution**: 
- Upgraded to `facebook/esm2_t33_650M_UR50D` (33 layers, 650M params)
- Added LoRA (Low-Rank Adaptation) support for efficient fine-tuning
- Only trains ~1-2% of parameters (vs 100% full fine-tuning)

**Configuration**:
```python
USE_650M_MODEL = True  # Set to False for 35M baseline
USE_LORA = True  # Efficient fine-tuning (requires: pip install peft)
```

**Benefits**:
- Better long-range dependency modeling
- Faster training than full fine-tuning
- Lower memory footprint
- Prevents catastrophic forgetting

### 2. **Sequence Length Fix: 512 → 1024**
**Problem**: ESM-2 supports 1024 tokens, but we were only using 512
**Critical Issue**: Mutations beyond position 512 were invisible to the model

**Solution**:
```python
MAX_LEN = 1024  # ESM-2's actual limit (was 512)
```

**Added Validation**:
- Automatically detects sequences > 1024 AA
- Warns about truncation risks
- Shows distribution statistics

**Impact**: 
- Can now process proteins up to 1024 amino acids
- Prevents silent failures from truncation

### 3. **Balanced Batch Sampling**
**Problem**: Imbalanced datasets (e.g., ClinVar with 10% toxic) cause "lazy guessing"
**Solution**: Added `WeightedRandomSampler` for balanced batches

**Configuration**:
```python
USE_BALANCED_SAMPLER = False  # Set to True for imbalanced data
```

**How it works**:
- Calculates inverse frequency weights
- Ensures 50/50 Safe/Toxic ratio in every batch
- Allows oversampling rare class (with replacement)

**When to use**:
- ClinVar data (likely imbalanced)
- Any dataset with > 2:1 class ratio
- When you see high accuracy but low MCC/recall

### 4. **Sequence Length Validation**
**New Feature**: Automatic detection and reporting of problematic sequences

**Output Example**:
```
⚠️  WARNING: 1,234 sequences exceed 1024 amino acids
   Max length: 3,418
   These will be truncated and may lose critical mutations!
   
   Sequence length distribution:
   < 512 AA: 1,500,000
   512-1024 AA: 800,000
   > 1024 AA: 1,234
```

**Action Items**:
- Filter very long sequences (> 1024 AA)
- Use sliding windows for long proteins
- Consider alternative architectures for very long sequences

## 📊 Configuration Guide

### For Current ProteinGym Dataset (Balanced)
```python
USE_650M_MODEL = True
USE_LORA = True
USE_BALANCED_SAMPLER = False  # Already balanced
MAX_LEN = 1024
BATCH_SIZE = 64  # For 650M
```

### For Future ClinVar Dataset (Imbalanced)
```python
USE_650M_MODEL = True
USE_LORA = True
USE_BALANCED_SAMPLER = True  # Enable for imbalanced data
MAX_LEN = 1024
BATCH_SIZE = 64
```

### For Quick Baseline (35M)
```python
USE_650M_MODEL = False
USE_LORA = False
USE_BALANCED_SAMPLER = False
MAX_LEN = 1024
BATCH_SIZE = 128
```

## 🔧 Installation Requirements

### Required
```bash
pip install torch transformers pandas numpy scikit-learn tqdm
```

### Optional (for LoRA)
```bash
pip install peft
```

**Note**: Script will fall back to manual fine-tuning if PEFT is not installed.

## 📈 Expected Improvements

### Performance
- **Better MCC**: 650M model should achieve 0.20+ MCC (vs 0.13 with 35M)
- **Better Recall**: Balanced sampling prevents "all safe" predictions
- **Longer Sequences**: Can handle proteins up to 1024 AA

### Training Efficiency
- **LoRA**: Trains ~10x faster than full fine-tuning
- **Memory**: Uses ~60% less GPU memory than full fine-tuning
- **Stability**: Less prone to catastrophic forgetting

## ⚠️ Known Limitations

### 1. Homology Leakage
**Status**: Not yet addressed
**Risk**: Model may memorize protein families instead of learning physics
**Mitigation**: Monitor per-family performance, consider homology-based splitting later

### 2. Very Long Sequences (> 1024 AA)
**Status**: Detected but not fully handled
**Risk**: Mutations beyond position 1024 are lost
**Solutions**:
- Filter out very long sequences
- Use sliding windows
- Consider ESM-2 variants with longer context

### 3. ClinVar Processing
**Status**: Script provided but needs testing
**Issues**: 
- Requires amino acid conversion (3-letter → 1-letter)
- Needs proper position parsing
- Must handle isoform mismatches

## 🎯 Next Steps

1. **Test with Current Data**: Run training with 650M + LoRA on ProteinGym
2. **Monitor Metrics**: Watch for MCC improvement (target: >0.20)
3. **Add ClinVar**: Integrate ClinVar data with balanced sampling
4. **Homology Analysis**: Add per-family performance tracking
5. **Long Sequence Handling**: Implement sliding windows or filtering

## 📝 Migration Notes

### From V1 to V2
- **No breaking changes**: Script is backward compatible
- **New defaults**: 650M model enabled by default
- **New warnings**: Sequence length validation may show warnings
- **Optional features**: LoRA and balanced sampling can be disabled

### Backward Compatibility
- Set `USE_650M_MODEL = False` to use old 35M model
- Set `USE_LORA = False` to use manual fine-tuning
- Set `USE_BALANCED_SAMPLER = False` for balanced datasets

## 🔍 Debugging Tips

### If training is slow:
- Reduce `BATCH_SIZE` (try 32)
- Increase `GRADIENT_ACCUMULATION_STEPS` (try 4)
- Check GPU memory usage

### If MCC is low:
- Enable `USE_BALANCED_SAMPLER` if data is imbalanced
- Check sequence length warnings
- Verify labels are correct (0/1)

### If out of memory:
- Reduce `BATCH_SIZE` to 32 or 16
- Set `USE_MIXED_PRECISION = True`
- Use `GRADIENT_ACCUMULATION_STEPS = 4`

## 📚 References

- **ESM-2 Paper**: [Lin et al., 2022](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1)
- **LoRA Paper**: [Hu et al., 2021](https://arxiv.org/abs/2106.09685)
- **PEFT Library**: https://github.com/huggingface/peft

