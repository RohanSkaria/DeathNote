#!/usr/bin/env python3
"""
train_colab.py - Streamlined training script for Google Colab

Death Note 2026 Phase 2: Train ESM-2 classifier on balanced dataset

Usage in Colab:
1. Upload train_balanced.csv and test_balanced.csv to /content/
2. Run this script

Expected columns: sequence_id, full_sequence, source, failure_type, label, score
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef
import numpy as np

# ============================================================
# CONFIGURATION - Adjust for your Colab environment
# ============================================================

# Model
MODEL_NAME = "facebook/esm2_t33_650M_UR50D"  # 650M parameter ESM-2

# Training
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 2e-5
MAX_LEN = 1024
GRADIENT_ACCUMULATION_STEPS = 4  # Effective batch: 8*4=32

# Memory optimization
USE_MIXED_PRECISION = True
USE_GRADIENT_CHECKPOINTING = True

# Early stopping
EARLY_STOPPING_PATIENCE = 2

# Data paths (Colab)
TRAIN_PATH = "train_balanced.csv"
TEST_PATH = "test_balanced.csv"

# Try to find data in common locations
for prefix in ["/content/", "/content/drive/MyDrive/DeathNote/", "./data/deathnote_data/"]:
    if os.path.exists(prefix + "train_balanced.csv"):
        TRAIN_PATH = prefix + "train_balanced.csv"
        TEST_PATH = prefix + "test_balanced.csv"
        break

# Output
SAVE_PATH = "/content/drive/MyDrive/DeathNote/models" if os.path.exists("/content/drive") else "./models"
os.makedirs(SAVE_PATH, exist_ok=True)

# ============================================================
# SETUP
# ============================================================

print("=" * 60)
print("Death Note 2026 - Phase 2 Training")
print("=" * 60)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
else:
    print("WARNING: No GPU detected!")

print(f"Train data: {TRAIN_PATH}")
print(f"Test data: {TEST_PATH}")
print(f"Model: {MODEL_NAME}")
print(f"Batch size: {BATCH_SIZE} x {GRADIENT_ACCUMULATION_STEPS} = {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")


# ============================================================
# DATASET (Pre-tokenized for efficiency)
# ============================================================

class ProteinDataset(Dataset):
    """Pre-tokenizes all sequences at init time to avoid per-batch tokenization overhead."""

    def __init__(self, df, tokenizer, max_len):
        self.labels = torch.tensor(df['label'].values, dtype=torch.float)

        # Track source for sample weighting (wild-types vs high-fitness)
        self.sources = df['source'].values if 'source' in df.columns else None

        # Clean sequences once upfront
        sequences = [
            str(s).strip() if s and len(str(s).strip()) > 0 else "M"
            for s in df['full_sequence'].values
        ]

        # Pre-tokenize all sequences (one-time cost)
        print(f"   Pre-tokenizing {len(sequences):,} sequences...")
        encodings = tokenizer(
            sequences,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        self.input_ids = encodings['input_ids']
        self.attention_mask = encodings['attention_mask']
        print(f"   Done. Shape: {self.input_ids.shape}")

    def get_sample_weights(self):
        """Compute sample weights to balance sources (upweight rare wild-types)."""
        if self.sources is None:
            return None

        # Count samples per source
        source_counts = {}
        for s in self.sources:
            source_counts[s] = source_counts.get(s, 0) + 1

        # Weight = 1 / count (rare sources get higher weight)
        total = len(self.sources)
        weights = []
        for s in self.sources:
            weights.append(total / source_counts[s])

        print(f"   Source distribution: {source_counts}")
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_mask[idx],
            'labels': self.labels[idx]
        }


# ============================================================
# MODEL
# ============================================================

class SafetyClassifier(nn.Module):
    def __init__(self, model_name, use_gradient_checkpointing=True):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

        if use_gradient_checkpointing:
            self.encoder.gradient_checkpointing_enable()

        # Freeze all but last layer
        for param in self.encoder.parameters():
            param.requires_grad = False

        # Unfreeze last transformer layer
        for param in self.encoder.encoder.layer[-1].parameters():
            param.requires_grad = True

        # Classification head
        hidden_size = self.encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling
        last_hidden = outputs.last_hidden_state
        mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        sum_embeddings = torch.sum(last_hidden * mask_expanded, dim=1)
        sum_mask = torch.clamp(mask_expanded.sum(dim=1), min=1e-9)
        pooled = sum_embeddings / sum_mask
        return self.classifier(pooled).squeeze(-1)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, dataloader, optimizer, criterion, scaler, device, accumulation_steps):
    model.train()
    total_loss = 0
    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc="Training")
    for i, batch in enumerate(pbar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels) / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pbar.set_postfix({'loss': f'{loss.item() * accumulation_steps:.4f}'})

    return total_loss / len(dataloader)


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=USE_MIXED_PRECISION):
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)

            total_loss += loss.item()
            preds = (torch.sigmoid(outputs) > 0.5).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Metrics
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mcc = matthews_corrcoef(all_labels, all_preds)

    return {
        'loss': total_loss / len(dataloader),
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1': f1,
        'mcc': mcc
    }


# ============================================================
# MAIN
# ============================================================

def main():
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    # Filter long sequences
    train_df = train_df[train_df['full_sequence'].str.len() <= MAX_LEN]
    test_df = test_df[test_df['full_sequence'].str.len() <= MAX_LEN]

    print(f"Train: {len(train_df):,} sequences")
    print(f"Test: {len(test_df):,} sequences")

    # Tokenizer
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Datasets
    train_dataset = ProteinDataset(train_df, tokenizer, MAX_LEN)
    test_dataset = ProteinDataset(test_df, tokenizer, MAX_LEN)

    # Use weighted sampling to balance sources (upweight rare wild-types)
    sample_weights = train_dataset.get_sample_weights()
    if sample_weights is not None:
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=2)
        print("   Using weighted sampling to balance sources")
    else:
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # Model
    print("\nLoading model...")
    model = SafetyClassifier(MODEL_NAME, USE_GRADIENT_CHECKPOINTING).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    # Training setup
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE)
    criterion = nn.BCEWithLogitsLoss()
    scaler = torch.cuda.amp.GradScaler(enabled=USE_MIXED_PRECISION)

    # Training loop
    best_f1 = 0
    patience_counter = 0

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")

        train_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device, GRADIENT_ACCUMULATION_STEPS)
        print(f"Train Loss: {train_loss:.4f}")

        metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1: {metrics['f1']:.4f}")
        print(f"MCC: {metrics['mcc']:.4f}")

        # Save best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            patience_counter = 0
            save_path = os.path.join(SAVE_PATH, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model (F1: {best_f1:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    print("\n" + "=" * 60)
    print(f"Training Complete! Best F1: {best_f1:.4f}")
    print(f"Model saved to: {SAVE_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
