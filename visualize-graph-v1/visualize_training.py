"""
Visualization script for training results
Creates comprehensive plots showing model performance over epochs
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Check if file exists
csv_path = "training_history.csv"
if not os.path.exists(csv_path):
    print(f"❌ Error: {csv_path} not found!")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in directory: {os.listdir('.')}")
    exit(1)

# Load Data
print(f"📊 Loading {csv_path}...")
df = pd.read_csv(csv_path)

# Validate columns
required_cols = ['epoch', 'train_loss', 'val_loss', 'mcc', 'recall']
missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    print(f"❌ Error: Missing columns: {missing_cols}")
    print(f"   Available columns: {list(df.columns)}")
    exit(1)

print(f"✅ Loaded {len(df)} epochs of training data")
print(f"   Columns: {list(df.columns)}")

# Setup Plot Style
sns.set_theme(style="whitegrid", palette="muted")
fig = plt.figure(figsize=(16, 10))

# --- Subplot 1: Loss Curves ---
plt.subplot(2, 3, 1)
plt.plot(df['epoch'], df['train_loss'], marker='o', label='Training Loss', 
         linestyle='--', linewidth=2, markersize=8, alpha=0.7)
plt.plot(df['epoch'], df['val_loss'], marker='s', label='Validation Loss', 
         linewidth=3, markersize=8, color='red')
plt.title("Learning Curve: Loss Over Time", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Loss (BCE)", fontsize=12)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# --- Subplot 2: Accuracy Curves ---
plt.subplot(2, 3, 2)
if 'train_acc' in df.columns and 'val_acc' in df.columns:
    plt.plot(df['epoch'], df['train_acc'], marker='o', label='Training Accuracy', 
             linestyle='--', linewidth=2, markersize=8, alpha=0.7)
    plt.plot(df['epoch'], df['val_acc'], marker='s', label='Validation Accuracy', 
             linewidth=3, markersize=8, color='green')
    plt.title("Accuracy Over Time", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.0)

# --- Subplot 3: MCC (Key Metric) ---
plt.subplot(2, 3, 3)
plt.plot(df['epoch'], df['mcc'], marker='d', color='purple', label='MCC', 
         linewidth=3, markersize=10)
plt.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, 
            label='Industry Standard (0.6)', linewidth=2)
plt.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, 
            label='Good (0.5)', linewidth=1)
plt.title("Matthews Correlation Coefficient", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("MCC", fontsize=12)
plt.ylim(0, 1.0)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# --- Subplot 4: Recall (Safety Metric) ---
plt.subplot(2, 3, 4)
plt.plot(df['epoch'], df['recall'], marker='s', color='green', label='Recall', 
         linewidth=3, markersize=10)
plt.axhline(y=0.8, color='gray', linestyle='--', alpha=0.5, 
            label='High Recall Target (0.8)', linewidth=2)
plt.title("Recall (Safety Detection)", fontsize=14, fontweight='bold')
plt.xlabel("Epoch", fontsize=12)
plt.ylabel("Recall", fontsize=12)
plt.ylim(0, 1.0)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)

# --- Subplot 5: Precision ---
plt.subplot(2, 3, 5)
if 'precision' in df.columns:
    plt.plot(df['epoch'], df['precision'], marker='^', color='blue', label='Precision', 
             linewidth=3, markersize=10)
    plt.title("Precision", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Precision", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

# --- Subplot 6: F1 Score ---
plt.subplot(2, 3, 6)
if 'f1' in df.columns:
    plt.plot(df['epoch'], df['f1'], marker='*', color='orange', label='F1 Score', 
             linewidth=3, markersize=10)
    plt.title("F1 Score (Balanced Metric)", fontsize=14, fontweight='bold')
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("F1 Score", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

# Add overall title
fig.suptitle("Model Training Performance Dashboard", fontsize=16, fontweight='bold', y=0.995)

# Final metrics summary
best_epoch = df.loc[df['mcc'].idxmax()]
print(f"\n📈 Best Performance (Epoch {int(best_epoch['epoch'])}):")
print(f"   MCC: {best_epoch['mcc']:.4f}")
print(f"   Recall: {best_epoch['recall']:.4f}")
if 'precision' in df.columns:
    print(f"   Precision: {best_epoch['precision']:.4f}")
if 'f1' in df.columns:
    print(f"   F1: {best_epoch['f1']:.4f}")
print(f"   Val Loss: {best_epoch['val_loss']:.4f}")

plt.tight_layout()
output_file = "results_graph.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✅ Graph saved to {output_file}")

# Also create a simple 2-panel version
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss
ax1.plot(df['epoch'], df['train_loss'], marker='o', label='Training Loss', 
         linestyle='--', linewidth=2, alpha=0.7)
ax1.plot(df['epoch'], df['val_loss'], marker='o', label='Validation Loss', 
         linewidth=3, color='red')
ax1.set_title("Learning Curve: Loss Decreases", fontsize=14, fontweight='bold')
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss (Error)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Key Metrics
ax2.plot(df['epoch'], df['recall'], marker='s', color='green', label='Recall (Safety)', 
         linewidth=2, markersize=8)
ax2.plot(df['epoch'], df['mcc'], marker='d', color='purple', label='MCC (Intelligence)', 
         linewidth=2, markersize=8)
ax2.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, 
            label='Industry Standard (0.6)', linewidth=2)
ax2.set_title("Model Intelligence: Hitting Industry Standards", fontsize=14, fontweight='bold')
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Score")
ax2.set_ylim(0, 1.0)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_file2 = "results_graph_simple.png"
plt.savefig(output_file2, dpi=300, bbox_inches='tight')
print(f"✅ Simple graph saved to {output_file2}")

