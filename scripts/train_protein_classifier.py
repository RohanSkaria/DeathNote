"""
Improved Protein Safety Classifier Training Script for Colab
Fixes and enhancements:
- Proper learning rate scheduler configuration
- Mixed precision training for A100
- Better metrics (precision, recall, F1)
- Early stopping
- Model checkpointing
- Gradient accumulation option
- Better error handling
- 650M model with LoRA support
- Sequence length validation (1024 token limit)
- Balanced batch sampling for imbalanced datasets
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import numpy as np
import glob
from datetime import datetime

# Try to import PEFT for LoRA (optional - falls back to manual fine-tuning if not available)
try:
    from peft import LoraConfig, get_peft_model, TaskType
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("⚠️  PEFT not installed.")
    print("   For Colab, run: !pip install peft")
    print("   Falling back to manual fine-tuning strategy.")

# --- CONFIGURATION ---
# 650M Model Configuration (No 35M fallback)
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 33 layers, 650M params
USE_LORA = True  # Use LoRA for efficient fine-tuning

BATCH_SIZE = 8  # Reduced for memory safety (650M + 1024 tokens is memory intensive)
EPOCHS = 1
LEARNING_RATE = 3e-4  # Lowered for fine-tuning (was 1e-4 for linear probe)
MAX_LEN = 1024  # CRITICAL FIX: ESM-2 supports 1024 tokens (was 512) - Required for pdLLT proteins
GRADIENT_ACCUMULATION_STEPS = 2  # Effective batch size: 8*2=16
FILTER_LONG_SEQUENCES = True  # Filter sequences > MAX_LEN to prevent OOM
USE_GRADIENT_CHECKPOINTING = True  # Re-enabled for memory safety with 1024 tokens
USE_MIXED_PRECISION = True  # Enable for A100 (faster training)
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N epochs
FINE_TUNE_LAST_LAYER = True  # Unfreeze last transformer layer (fallback if no LoRA)
USE_BALANCED_SAMPLER = True  # Set to True if dataset is imbalanced (e.g., ClinVar)

# --- AUTO-DETECT DATA PATH (Colab-friendly) ---
# Colab paths: /content/ for uploaded files, /content/drive/MyDrive/ for Google Drive
# Force use of ClinVar training data
DATA_PATH = "clinvar_training_data.csv"

# Check if file exists in common locations
if not os.path.exists(DATA_PATH):
    # Try Colab paths
    colab_paths = [
        f"/content/data/{DATA_PATH}",
        f"/content/drive/MyDrive/Death Note/{DATA_PATH}",
        f"/content/{DATA_PATH}"
    ]
    for path in colab_paths:
        if os.path.exists(path):
            DATA_PATH = path
            break
    else:
        # Fallback to current directory or default
        if not os.path.exists(DATA_PATH):
            print(f"⚠️  {DATA_PATH} not found. Checking other locations...")
            possible_files = glob.glob("*.csv")
            if possible_files:
                DATA_PATH = possible_files[0]
                print(f"✅ Using found CSV file: {DATA_PATH}")
            else:
                print(f"⚠️  Using default path (file may not exist): {DATA_PATH}")

print(f"✅ Using ClinVar training data: {DATA_PATH}")

SAVE_PATH = '/content/drive/MyDrive/Death Note/models'
os.makedirs(SAVE_PATH, exist_ok=True)

# Colab GPU detection
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"🚀 Training on: {gpu_name} ({gpu_memory:.1f} GB)")
    print(f"   CUDA Version: {torch.version.cuda}")
else:
    print("⚠️  No GPU detected! Training will be very slow on CPU.")
    print("   In Colab: Runtime → Change runtime type → GPU (T4/A100)")

print(f"📂 Reading data from: {DATA_PATH}")
print(f"💾 Saving models to: {SAVE_PATH}")
print(f"🔧 Mixed Precision: {USE_MIXED_PRECISION}")
print(f"🔧 Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS} (effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS})")
print(f"🔧 Model: {MODEL_NAME} (650M)")
print(f"🔧 LoRA: {'Enabled' if (USE_LORA and HAS_PEFT) else 'Disabled'}")
print(f"🔧 Filter Long Sequences: {FILTER_LONG_SEQUENCES}")
print(f"🔧 Batch Size: {BATCH_SIZE}")

# --- PRE-TOKENIZATION (Batch Mode for Speed) ---
def prepare_fast_data(df, tokenizer, max_len, filename="data_cache.pt"):
    """
    Pre-tokenize all sequences in batch mode for massive speedup.
    Saves to local disk (/content/) for fast loading.
    """
    print(f"⏳ Pre-tokenizing {len(df):,} sequences (Batch Mode)...")
    encodings = tokenizer(
        df['full_sequence'].tolist(),
        truncation=True,
        padding='max_length',
        max_length=max_len,
        return_tensors='pt'
    )
    data = {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': torch.tensor(df['label'].values, dtype=torch.float)
    }
    torch.save(data, filename)
    print(f"✅ Cache saved to {filename}")
    return filename

# --- FAST DATASET (Loads Pre-Tokenized Cache) ---
class FastProteinDataset(Dataset):
    """Dataset that loads pre-tokenized data from cache - eliminates tokenization bottleneck"""
    def __init__(self, cache_file):
        # Load cache once - keep tensors on CPU, move to GPU in DataLoader
        data = torch.load(cache_file, map_location='cpu')
        self.input_ids = data['input_ids']
        self.attention_mask = data['attention_mask']
        self.labels = data['labels']
        # Expose labels as numpy array for balanced sampling
        self.labels_numpy = self.labels.numpy()
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        # Direct tensor indexing - very fast
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }

# --- LEGACY DATASET (Kept for backward compatibility) ---
class ProteinDataset(Dataset):
    """Original dataset with on-the-fly tokenization (slow, kept for fallback)"""
    def __init__(self, df, tokenizer, max_len):
        self.sequences = df['full_sequence'].values
        self.labels = df['label'].values
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        seq = str(self.sequences[item])
        label = self.labels[item]

        # Handle empty sequences
        if not seq or len(seq.strip()) == 0:
            seq = "M"  # Default to methionine if empty

        encoding = self.tokenizer.encode_plus(
            seq,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# --- MODEL ---
class SafetyClassifier(nn.Module):
    def __init__(self, model_name, fine_tune_last_layer=True, use_lora=False, use_gradient_checkpointing=True):
        super(SafetyClassifier, self).__init__()

        # Load base model
        # NOTE: Keep in FP32 - GradScaler needs FP32 weights for safe mixed precision training
        # Autocast in training loop will handle FP16 computation automatically
        base_model = AutoModel.from_pretrained(model_name)

        # Enable gradient checkpointing to save memory (trades compute for memory)
        # This is critical for 650M model on A100
        if use_gradient_checkpointing and hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
            print("✅ Enabled gradient checkpointing (memory optimization)")

        # Strategy 1: Use LoRA (if available and requested)
        if use_lora and HAS_PEFT:
            print("🔧 Using LoRA for parameter-efficient fine-tuning...")
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=False,
                r=16,  # Rank: Higher = more capacity but slower
                lora_alpha=32,  # Scaling factor
                lora_dropout=0.05,
                target_modules=["query", "key", "value", "dense"]  # Fine-tune attention + dense layers
            )
            self.esm = get_peft_model(base_model, peft_config)
            self.esm.print_trainable_parameters()

        # Strategy 2: Manual fine-tuning (fallback if LoRA not available)
        else:
            self.esm = base_model

            # FREEZE most of the model
            for param in self.esm.parameters():
                param.requires_grad = False

            # UNFREEZE the last Transformer Layer (Fine-Tuning Strategy)
            if fine_tune_last_layer and hasattr(self.esm, 'encoder') and hasattr(self.esm.encoder, 'layer'):
                for param in self.esm.encoder.layer[-1].parameters():
                    param.requires_grad = True
                print("✅ Unfrozen last transformer layer for fine-tuning")

            # Also unfreeze the pooler if it exists
            if hasattr(self.esm, 'pooler') and self.esm.pooler is not None:
                for param in self.esm.pooler.parameters():
                    param.requires_grad = True

            print("⚠️  LoRA not available - using manual fine-tuning (slower, uses more memory)")

        # UPGRADED Classifier Head (Add Hidden Layer)
        # hidden_size -> 256 -> 1 (Instead of just hidden_size -> 1)
        hidden_size = base_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Use CLS token (index 0)
        pooled_output = output.last_hidden_state[:, 0, :]

        # Pass through the upgraded classifier head
        return self.classifier(pooled_output)

# --- TRAINING ENGINE ---
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples,
                use_amp=False, scaler=None, grad_accum_steps=1):
    model = model.train()
    losses = []
    correct_predictions = 0

    progress_bar = tqdm(data_loader, desc="Training", unit="batch")
    optimizer.zero_grad()  # Initialize gradients

    for batch_idx, d in enumerate(progress_bar):
        # Move to device (non-blocking for better pipelining)
        input_ids = d["input_ids"].to(device, non_blocking=True)
        attention_mask = d["attention_mask"].to(device, non_blocking=True)
        labels = d["labels"].to(device, non_blocking=True)

        # Mixed precision forward pass (Fixed deprecated API)
        if use_amp:
            with torch.amp.autocast('cuda'):  # Fixed: use torch.amp instead of torch.cuda.amp
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs).flatten()
                loss = loss_fn(outputs.flatten(), labels)
                loss = loss / grad_accum_steps  # Scale loss for gradient accumulation
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.sigmoid(outputs).flatten()
            loss = loss_fn(outputs.flatten(), labels)
            loss = loss / grad_accum_steps

        losses.append(loss.item() * grad_accum_steps)  # Unscale for logging

        predicted_labels = (preds > 0.5).float()
        correct_predictions += torch.sum(predicted_labels == labels)

        # Backward pass
        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation: only step optimizer every N batches
        if (batch_idx + 1) % grad_accum_steps == 0:
            # Clear cache periodically during training to prevent fragmentation
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            if use_amp:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()

        progress_bar.set_postfix(loss=np.mean(losses))

    return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples, use_amp=False):
    model = model.eval()
    losses = []
    correct_predictions = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for d in tqdm(data_loader, desc="Evaluating"):
            # Move to device (non-blocking for better pipelining)
            input_ids = d["input_ids"].to(device, non_blocking=True)
            attention_mask = d["attention_mask"].to(device, non_blocking=True)
            labels = d["labels"].to(device, non_blocking=True)

            if use_amp:
                with torch.amp.autocast('cuda'):  # Fixed: use torch.amp instead of torch.cuda.amp
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    preds = torch.sigmoid(outputs).flatten()
                    loss = loss_fn(outputs.flatten(), labels)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.sigmoid(outputs).flatten()
                loss = loss_fn(outputs.flatten(), labels)

            losses.append(loss.item())

            predicted_labels = (preds > 0.5).float()
            correct_predictions += torch.sum(predicted_labels == labels)

            all_preds.extend(predicted_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate comprehensive metrics
    accuracy = correct_predictions.double() / n_examples
    avg_loss = np.mean(losses)
    mcc = matthews_corrcoef(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)

    return accuracy, avg_loss, {
        'mcc': mcc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# --- MAIN ---
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Enable TF32 for faster matmul on A100 (Tensor Core acceleration)
    # This gives speedup without torch.compile() overhead
    if torch.cuda.is_available():
        # Use new API to avoid deprecation warnings
        torch.backends.cuda.matmul.fp32_precision = 'tf32'
        torch.backends.cudnn.allow_tf32 = True
        print("✅ Enabled TF32 acceleration (Tensor Cores)")

    # 1. LOAD THE SMOKE TEST DATA (Gene-Split to prevent homology leakage)
    print("\n📊 Loading Smoke Test Splits...")
    print("   Using pre-split data to ensure NO gene overlap between train/val")
    
    # Try multiple locations for Colab compatibility
    train_paths = [
        "train_smoke_test.csv",
        "/content/data/train_smoke_test.csv",
        "/content/drive/MyDrive/Death Note/train_smoke_test.csv",
        "/content/train_smoke_test.csv"
    ]
    val_paths = [
        "val_smoke_test.csv",
        "/content/data/val_smoke_test.csv",
        "/content/drive/MyDrive/Death Note/val_smoke_test.csv",
        "/content/val_smoke_test.csv"
    ]
    
    train_path = None
    val_path = None
    
    for path in train_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    for path in val_paths:
        if os.path.exists(path):
            val_path = path
            break
    
    if train_path is None or val_path is None:
        print("❌ ERROR: Smoke test files not found!")
        print("   Expected files: train_smoke_test.csv and val_smoke_test.csv")
        print("   Please run the data splitter script first to generate these files.")
        raise FileNotFoundError("Smoke test CSV files not found")
    
    # Load the splits
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print(f"✅ Loaded smoke test splits:")
    print(f"   Train: {len(train_df):,} samples from {train_path}")
    print(f"   Val:   {len(val_df):,} samples from {val_path}")
    
    # Validate data structure
    for df_name, df in [("Train", train_df), ("Val", val_df)]:
        if 'full_sequence' not in df.columns or 'label' not in df.columns:
            raise ValueError(f"{df_name} CSV must contain 'full_sequence' and 'label' columns")
        
        # Filter invalid rows
        initial_len = len(df)
        df = df[df['full_sequence'].notna()].copy()
        df = df[df['full_sequence'].str.len() > 0].copy()
        df = df[df['label'].isin([0, 1])].copy()
        
        if len(df) < initial_len:
            print(f"   ⚠️  Filtered out {initial_len - len(df)} invalid rows from {df_name}")
        
        # Update the dataframe
        if df_name == "Train":
            train_df = df
        else:
            val_df = df
    
    # Validate sequence lengths
    print("\n🔍 Validating sequence lengths...")
    for df_name, df in [("Train", train_df), ("Val", val_df)]:
        df['seq_length'] = df['full_sequence'].str.len()
        long_seqs = df[df['seq_length'] > MAX_LEN]
        
        if len(long_seqs) > 0:
            if FILTER_LONG_SEQUENCES:
                initial_count = len(df)
                df = df[df['seq_length'] <= MAX_LEN].copy()
                filtered_count = initial_count - len(df)
                print(f"   {df_name}: Filtered out {filtered_count:,} sequences > {MAX_LEN} AA")
                if df_name == "Train":
                    train_df = df
                else:
                    val_df = df
            else:
                print(f"   ⚠️  {df_name}: {len(long_seqs):,} sequences will be truncated")
    
    # Print statistics
    print(f"\n📊 Dataset Statistics:")
    print(f"   Train - Total: {len(train_df):,}")
    print(f"      Label 0: {(train_df['label'] == 0).sum():,} ({(train_df['label'] == 0).sum()/len(train_df)*100:.1f}%)")
    print(f"      Label 1: {(train_df['label'] == 1).sum():,} ({(train_df['label'] == 1).sum()/len(train_df)*100:.1f}%)")
    print(f"   Val   - Total: {len(val_df):,}")
    print(f"      Label 0: {(val_df['label'] == 0).sum():,} ({(val_df['label'] == 0).sum()/len(val_df)*100:.1f}%)")
    print(f"      Label 1: {(val_df['label'] == 1).sum():,} ({(val_df['label'] == 1).sum()/len(val_df)*100:.1f}%)")
    
    # 2. TOKENIZER & PRE-TOKENIZATION
    print("\n🔤 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Check for pre-tokenized cache (fast path)
    cache_dir = "/content"  # Use local disk, not Drive (faster I/O)
    os.makedirs(cache_dir, exist_ok=True)
    
    train_cache = os.path.join(cache_dir, "data_cache_train.pt")
    val_cache = os.path.join(cache_dir, "data_cache_val.pt")
    
    # Pre-tokenize if cache doesn't exist
    if not os.path.exists(train_cache) or not os.path.exists(val_cache):
        print("📦 Pre-tokenizing datasets (one-time operation)...")
        prepare_fast_data(train_df, tokenizer, MAX_LEN, train_cache)
        prepare_fast_data(val_df, tokenizer, MAX_LEN, val_cache)
    else:
        print(f"✅ Using cached tokenized data from {cache_dir}")
    
    # Load fast datasets
    train_dataset = FastProteinDataset(train_cache)
    val_dataset = FastProteinDataset(val_cache)

    # 3. BALANCED SAMPLING (always enabled for smoke test data)
    print("\n⚖️ Using WeightedRandomSampler for balanced batches...")
    labels = train_dataset.labels_numpy.astype(int)
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[labels]
    
    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print(f"   Class weights: {dict(zip(range(len(class_weights)), class_weights))}")

    # 4. DATALOADERS (Optimized for Colab - num_workers=0 eliminates multiprocessing overhead)
    # With pre-tokenized data, num_workers=0 is often faster in Colab due to no IPC overhead
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        sampler=train_sampler, 
        num_workers=0,  # Zero workers = no multiprocessing overhead (faster for pre-tokenized data)
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        num_workers=0,  # Zero workers for consistency
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Model Setup
    print(f"\n🤖 Loading Model: {MODEL_NAME}")
    print(f"   Using LoRA: {USE_LORA and HAS_PEFT}")
    print(f"   Max sequence length: {MAX_LEN} tokens")

    use_lora_flag = USE_LORA and HAS_PEFT
    model = SafetyClassifier(
        MODEL_NAME,
        fine_tune_last_layer=FINE_TUNE_LAST_LAYER,
        use_lora=use_lora_flag,
        use_gradient_checkpointing=USE_GRADIENT_CHECKPOINTING
    )

    # Move model to device with memory optimization
    # Clear cache before moving model
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        
        # Set memory allocation strategy to reduce fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

    # Move to device - keep in FP32 for GradScaler compatibility
    # Autocast in training loop will handle FP16 computation automatically
    model = model.to(device)
    
    # Show memory usage after model load
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU Memory after model load - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # DO NOT convert to .half() - GradScaler needs FP32 weights
    # The autocast block in train_epoch handles FP16 computation safely

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.2f}%)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # FIXED: Proper learning rate scheduler
    # LinearLR needs start_factor and end_factor, or use CosineAnnealingLR/OneCycleLR
    total_steps = len(train_loader) * EPOCHS // GRADIENT_ACCUMULATION_STEPS
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=LEARNING_RATE * 0.01
    )

    loss_fn = nn.BCEWithLogitsLoss().to(device)

    # Mixed precision scaler (Fixed deprecated API)
    if USE_MIXED_PRECISION:
        scaler = torch.amp.GradScaler('cuda')  # Fixed: use torch.amp instead of torch.cuda.amp
    else:
        scaler = None

    # Clear GPU cache before training and set memory allocation strategy
    if torch.cuda.is_available():
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        # Set memory allocation strategy to reduce fragmentation
        import os
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

        print(f"🧹 Cleared GPU cache and set memory allocation strategy")

        # Show current memory usage
        allocated = torch.cuda.memory_allocated(0) / 1e9
        reserved = torch.cuda.memory_reserved(0) / 1e9
        print(f"   GPU Memory - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

    # Training loop
    print(f"\n🚀 Starting Training ({EPOCHS} epochs)...")
    print("="*80)

    best_mcc = -1
    patience_counter = 0
    training_history = []

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print("-" * 80)

        train_acc, train_loss = train_epoch(
            model, train_loader, loss_fn, optimizer, device, scheduler,
            len(train_df), USE_MIXED_PRECISION, scaler, GRADIENT_ACCUMULATION_STEPS
        )
        print(f"Train - Loss: {train_loss:.4f} | Accuracy: {train_acc:.4f}")

        val_acc, val_loss, val_metrics = eval_model(
            model, val_loader, loss_fn, device, len(val_df), USE_MIXED_PRECISION
        )
        print(f"Val   - Loss: {val_loss:.4f} | Accuracy: {val_acc:.4f}")
        print(f"Val   - MCC: {val_metrics['mcc']:.4f} | Precision: {val_metrics['precision']:.4f} | "
              f"Recall: {val_metrics['recall']:.4f} | F1: {val_metrics['f1']:.4f}")

        # Save history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc.item(),
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            **val_metrics
        })

        # Save best model
        if val_metrics['mcc'] > best_mcc:
            print("🏆 New Best MCC! Saving model...")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_mcc': val_metrics['mcc'],
                'metrics': val_metrics,
            }, f"{SAVE_PATH}/best_model_state.bin")
            best_mcc = val_metrics['mcc']
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"\n⏹️  Early stopping triggered (no improvement for {EARLY_STOPPING_PATIENCE} epochs)")
                break

    # Save final model and training history
    print(f"\n💾 Saving final model and training history...")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_mcc': best_mcc,
    }, f"{SAVE_PATH}/final_model_state.bin")

    # Save training history as CSV
    history_df = pd.DataFrame(training_history)
    history_df.to_csv(f"{SAVE_PATH}/training_history.csv", index=False)
    print(f"✅ Training complete! Best MCC: {best_mcc:.4f}")
    print(f"📊 Training history saved to: {SAVE_PATH}/training_history.csv")

