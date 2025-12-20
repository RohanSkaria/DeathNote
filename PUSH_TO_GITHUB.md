# Push to GitHub - Git LFS Setup Guide

## Current Situation

Your repository has files > 100MB:
- `visualize-graph-v1/best_model_state.bin` (160MB)
- `data/processed/protein_data_batch_1.csv` (1.0GB)
- `data/processed/negative_data_batch_1.csv` (240MB)
- `data/processed/negative_data_batch_1_backup_20251220_113131.csv` (241MB)

**Total: ~1.6GB** - This exceeds GitHub's free LFS quota (1GB storage).

## Option 1: Use Git LFS (Recommended for Code + Small Model)

### Step 1: Install Git LFS

```bash
# macOS
brew install git-lfs

# Then initialize
git lfs install
```

### Step 2: Track Large Files

```bash
cd /Users/rohan/Development/DeathNote/DeathNote

# Track model file (160MB - fits in free tier)
git lfs track "*.bin"
git lfs track "visualize-graph-v1/best_model_state.bin"

# Add .gitattributes
git add .gitattributes
```

### Step 3: Exclude Large Data Files

**Recommendation:** Don't commit the CSV data files to GitHub. Instead:

1. Add data files to `.gitignore`:
```bash
echo "data/processed/*.csv" >> .gitignore
echo "data/processed/*_backup_*.csv" >> .gitignore
```

2. Or use DVC (Data Version Control) for data files
3. Or store data files in cloud storage (S3, Google Drive, etc.)

### Step 4: Commit and Push

```bash
# Add code files
git add scripts/
git add visualize-graph-v1/*.py
git add visualize-graph-v1/*.md
git add visualize-graph-v1/*.csv  # training_history.csv is small
git add visualize-graph-v1/best_model_state.bin  # Tracked via LFS
git add claude.md
git add .gitignore .gitattributes

# Commit
git commit -m "Add protein safety classifier with model checkpoint"

# Push
git push origin main
```

## Option 2: GitHub Releases (For Model Files)

Instead of committing the model file, upload it as a GitHub Release:

1. Go to your GitHub repo → Releases → Create a new release
2. Upload `best_model_state.bin` as a release asset
3. Update your code to download from releases if needed

## Option 3: Exclude Everything Large (Code Only)

Only commit code, keep data/model files local:

```bash
# Add to .gitignore
echo "data/" >> .gitignore
echo "visualize-graph-v1/*.bin" >> .gitignore
echo "DMS_ProteinGym_substitutions/" >> .gitignore

# Then commit only code
git add scripts/ visualize-graph-v1/*.py visualize-graph-v1/*.md
git add claude.md .gitignore
git commit -m "Add protein safety classifier code"
git push origin main
```

## Recommended Approach

**For your use case, I recommend:**

1. **Commit code** (scripts, app.py, etc.)
2. **Commit small files** (training_history.csv, README, etc.)
3. **Use Git LFS for model** (`best_model_state.bin` - 160MB)
4. **Exclude large CSV files** (add to .gitignore)
5. **Document where to get data** in README

This keeps your repo clean and under GitHub limits while preserving all code.

## Quick Commands

```bash
cd /Users/rohan/Development/DeathNote/DeathNote

# Install Git LFS (if not installed)
brew install git-lfs
git lfs install

# Track model file
git lfs track "*.bin"
git lfs track "visualize-graph-v1/best_model_state.bin"

# Update .gitignore to exclude large CSVs
cat >> .gitignore << EOF
# Large data files (use cloud storage or DVC)
data/processed/protein_data_batch_1.csv
data/processed/negative_data_batch_1.csv
data/processed/negative_data_batch_1_backup_*.csv
DMS_ProteinGym_substitutions/
EOF

# Add files
git add .gitattributes .gitignore
git add scripts/ visualize-graph-v1/*.py visualize-graph-v1/*.md visualize-graph-v1/training_history.csv
git add visualize-graph-v1/best_model_state.bin
git add claude.md

# Commit and push
git commit -m "Add protein safety classifier"
git push origin main
```

