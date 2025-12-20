# Model Upgrade Summary - Fixing Stuck Training

## Problem Identified

Training was getting stuck with minimal improvement:
- Epoch 1 MCC: 0.1303
- Epoch 2 MCC: 0.1284 (actually worse!)
- Model wasn't learning effectively

## Root Causes

1. **Fully Frozen Model**: Linear probe strategy (all layers frozen) limits adaptation
2. **Simple Classifier Head**: Single linear layer may not capture complex patterns
3. **Learning Rate Too High**: 1e-4 is too high for fine-tuning
4. **Deprecated API**: `torch.cuda.amp.autocast()` warnings

## Changes Applied

### 1. Fine-Tuning Strategy (Lines 89-120)
**Before**: All layers frozen (linear probe)
```python
for param in self.esm.parameters():
    param.requires_grad = False
```

**After**: Unfreeze last transformer layer
```python
# Freeze most layers
for param in self.esm.parameters():
    param.requires_grad = False

# Unfreeze last layer for fine-tuning
for param in self.esm.encoder.layer[-1].parameters():
    param.requires_grad = True
```

**Impact**: Model can now adapt features to your specific data while keeping most pretrained knowledge intact.

### 2. Deeper Classifier Head (Lines 108-115)
**Before**: Single linear layer
```python
self.drop = nn.Dropout(p=0.3)
self.out = nn.Linear(self.esm.config.hidden_size, 1)
```

**After**: Two-layer MLP with ReLU
```python
self.classifier = nn.Sequential(
    nn.Dropout(p=0.3),
    nn.Linear(self.esm.config.hidden_size, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 1)
)
```

**Impact**: More capacity to learn complex decision boundaries.

### 3. Lower Learning Rate (Line 19)
**Before**: `LEARNING_RATE = 1e-4`
**After**: `LEARNING_RATE = 5e-5`

**Impact**: More stable fine-tuning, prevents overwriting pretrained features too quickly.

### 4. Fixed Deprecated API (Lines 128, 182)
**Before**: `torch.cuda.amp.autocast()`
**After**: `torch.amp.autocast('cuda')`

**Impact**: Removes FutureWarning, uses current PyTorch API.

### 5. Added Configuration Flag (Line 22)
```python
FINE_TUNE_LAST_LAYER = True  # Can toggle fine-tuning on/off
```

**Impact**: Easy to switch between linear probe and fine-tuning strategies.

### 6. Parameter Count Logging (Lines 250-253)
Added logging to show trainable vs total parameters.

**Impact**: Helps understand model capacity and verify fine-tuning is working.

## Expected Improvements

1. **Better Convergence**: Fine-tuning should allow model to improve across epochs
2. **Higher Capacity**: Deeper head can learn more complex patterns
3. **Stable Training**: Lower LR prevents instability
4. **No Warnings**: Clean output without deprecation warnings

## Training Strategy Comparison

| Strategy | Frozen Layers | Trainable % | Learning Rate | Use Case |
|----------|--------------|-------------|---------------|----------|
| **Linear Probe** | All | ~0.1% | 1e-4 | Quick baseline |
| **Fine-Tune Last** | All but last | ~5-10% | 5e-5 | Better performance |
| **Full Fine-Tune** | None | 100% | 1e-5 | Best performance (slow) |

## Configuration

You can toggle fine-tuning:
```python
FINE_TUNE_LAST_LAYER = True   # Fine-tune (recommended)
FINE_TUNE_LAST_LAYER = False  # Linear probe (faster, less capacity)
```

## Next Steps

1. Run training with upgraded model
2. Monitor if MCC improves across epochs
3. If still stuck, consider:
   - Unfreezing more layers (last 2-3 layers)
   - Further lowering learning rate
   - Adding more regularization (higher dropout)
   - Trying different optimizer (Adam vs AdamW)

## Performance Expectations

- **Training Time**: Slightly slower (~5-10% due to unfrozen layer)
- **Memory**: Similar (only last layer gradients)
- **Accuracy**: Should improve significantly (MCC target: >0.15-0.20)
- **Convergence**: Should see steady improvement across epochs

