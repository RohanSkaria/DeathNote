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
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, matthews_corrcoef, precision_score, recall_score, f1_score
import numpy as np
import glob
from datetime import datetime

# --- CONFIGURATION ---
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"  # Fast baseline
BATCH_SIZE = 128  # Good for A100
EPOCHS = 3
LEARNING_RATE = 5e-5  # Lowered for fine-tuning (was 1e-4 for linear probe)
MAX_LEN = 512
GRADIENT_ACCUMULATION_STEPS = 1  # Increase for larger effective batch size
USE_MIXED_PRECISION = True  # Enable for A100 (faster training)
EARLY_STOPPING_PATIENCE = 3  # Stop if no improvement for N epochs
FINE_TUNE_LAST_LAYER = True  # Unfreeze last transformer layer for better adaptation

# --- AUTO-DETECT DATA PATH ---
possible_files = glob.glob("/content/data/*.csv")
if len(possible_files) > 0:
    DATA_PATH = possible_files[0]
else:
    # Fallback if you didn't run the unzip cell
    DATA_PATH = '/content/drive/MyDrive/Death Note/protein_data_batch_1.csv'

SAVE_PATH = '/content/drive/MyDrive/Death Note/models'
os.makedirs(SAVE_PATH, exist_ok=True)

print(f"🚀 Training on: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
print(f"📂 Reading data from: {DATA_PATH}")
print(f"💾 Saving models to: {SAVE_PATH}")
print(f"🔧 Mixed Precision: {USE_MIXED_PRECISION}")
print(f"🔧 Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")

# --- DATASET ---
class ProteinDataset(Dataset):
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
    def __init__(self, model_name, fine_tune_last_layer=True):
        super(SafetyClassifier, self).__init__()
        self.esm = AutoModel.from_pretrained(model_name)
        
        # FREEZE most of the model
        for param in self.esm.parameters():
            param.requires_grad = False
        
        # UNFREEZE the last Transformer Layer (Fine-Tuning Strategy)
        # This allows the model to adapt its features to your specific data
        if fine_tune_last_layer and hasattr(self.esm, 'encoder') and hasattr(self.esm.encoder, 'layer'):
            for param in self.esm.encoder.layer[-1].parameters():
                param.requires_grad = True
            print("✅ Unfrozen last transformer layer for fine-tuning")
        
        # Also unfreeze the pooler if it exists
        if hasattr(self.esm, 'pooler') and self.esm.pooler is not None:
            for param in self.esm.pooler.parameters():
                param.requires_grad = True
        
        # UPGRADED Classifier Head (Add Hidden Layer)
        # hidden_size -> 256 -> 1 (Instead of just hidden_size -> 1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.esm.config.hidden_size, 256),
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
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        labels = d["labels"].to(device)

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
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

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
    
    # Load Data
    print("\n📊 Loading CSV...")
    df = pd.read_csv(DATA_PATH)
    
    # Validate data
    if 'full_sequence' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'full_sequence' and 'label' columns")
    
    # Filter out any invalid rows
    initial_len = len(df)
    df = df[df['full_sequence'].notna()].copy()
    df = df[df['full_sequence'].str.len() > 0].copy()
    df = df[df['label'].isin([0, 1])].copy()
    
    if len(df) < initial_len:
        print(f"⚠️  Filtered out {initial_len - len(df)} invalid rows")
    
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
    
    print(f"✅ Loaded {len(df):,} samples")
    print(f"   Label 0: {(df['label'] == 0).sum():,} ({(df['label'] == 0).sum()/len(df)*100:.1f}%)")
    print(f"   Label 1: {(df['label'] == 1).sum():,} ({(df['label'] == 1).sum()/len(df)*100:.1f}%)")

    # Tokenizer
    print("\n🔤 Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Split (90/10)
    train_size = int(0.9 * len(df))
    val_size = len(df) - train_size
    train_df = df.iloc[:train_size]
    val_df = df.iloc[train_size:]
    
    print(f"\n📦 Dataset Split:")
    print(f"   Train: {len(train_df):,} samples")
    print(f"   Val:   {len(val_df):,} samples")

    # Data Loaders
    train_dataset = ProteinDataset(train_df, tokenizer, MAX_LEN)
    val_dataset = ProteinDataset(val_df, tokenizer, MAX_LEN)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=2, 
        pin_memory=True
    )

    # Model Setup
    print("\n🤖 Loading Model...")
    model = SafetyClassifier(MODEL_NAME, fine_tune_last_layer=FINE_TUNE_LAST_LAYER)
    model = model.to(device)
    
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
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if USE_MIXED_PRECISION else None

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

