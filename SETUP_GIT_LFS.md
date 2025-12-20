# Git LFS Setup for DeathNote Repository

## Large Files Detected

The following files exceed GitHub's 100MB limit:
- `visualize-graph-v1/best_model_state.bin` (160MB)
- `data/processed/protein_data_batch_1.csv` (1.0GB)
- `data/processed/negative_data_batch_1.csv` (240MB)
- `data/processed/negative_data_batch_1_backup_20251220_113131.csv` (241MB)

## Setup Instructions

### 1. Install Git LFS

**macOS:**
```bash
brew install git-lfs
```

**Linux:**
```bash
# Ubuntu/Debian
sudo apt-get install git-lfs

# Fedora
sudo dnf install git-lfs
```

**Windows:**
Download from: https://git-lfs.github.com/

### 2. Initialize Git LFS in Repository

```bash
cd /Users/rohan/Development/DeathNote/DeathNote
git lfs install
```

### 3. Track Large Files

The `.gitattributes` file is already configured. Verify it's working:

```bash
git lfs track
```

### 4. Add Files and Commit

```bash
# Add .gitattributes first
git add .gitattributes

# Add all files (LFS will handle large ones automatically)
git add .

# Commit
git commit -m "Initial commit with Git LFS for large files"

# Push (LFS files will be uploaded automatically)
git push origin main
```

## Verify LFS is Working

After pushing, verify files are tracked:

```bash
git lfs ls-files
```

You should see your large files listed.

## Important Notes

1. **GitHub LFS Limits:**
   - Free accounts: 1GB storage, 1GB bandwidth/month
   - Your files total ~1.6GB, so you may need a paid plan or use alternatives

2. **Alternative Options:**
   - Use GitHub Releases for model files
   - Store data files on cloud storage (S3, Google Drive, etc.)
   - Use DVC (Data Version Control) for data files
   - Only commit code, keep data/model files local or in cloud

3. **Recommended Approach:**
   - Commit code and small files to GitHub
   - Store large data/model files separately:
     - Model: Upload to GitHub Releases
     - Data: Store in cloud storage or use DVC

## Quick Setup Script

```bash
# Install Git LFS (if not installed)
brew install git-lfs  # macOS
# or: sudo apt-get install git-lfs  # Linux

# Initialize
cd /Users/rohan/Development/DeathNote/DeathNote
git lfs install

# Add and commit
git add .gitattributes .gitignore
git add scripts/ visualize-graph-v1/*.py visualize-graph-v1/*.csv visualize-graph-v1/*.md
git add claude.md

# For large files, add them explicitly
git add visualize-graph-v1/best_model_state.bin
# Note: Only add data files if you have GitHub LFS quota

git commit -m "Add protein safety classifier with Git LFS"
git push origin main
```

