# Death Note Project - Complete Summary

## Project Goal
Build a protein safety classifier using ESM-2 to predict whether protein sequences are "safe" (functional) or "toxic" (non-functional/harmful).

---

## Phase 1: Data Pipeline (Complete)

### Data Sources Ingested

| Source | Type | Count | Purpose |
|--------|------|-------|---------|
| ProteinGym Wild-Types | Positive | 160 | Canonical reference sequences |
| UniProt Human Reviewed | Positive | 9,939 | High-confidence wild-types |
| ProteinGym High-Fitness | Positive | 405,238 | DMS assay validated mutants |
| ProteinGym Low-Fitness | Negative | ~400K | Failed DMS assay mutants |
| ESOL Low-Yield | Negative | 2,318 | Low expression E. coli proteins |
| Aggregation-Prone | Negative | 2,143 | Aggregation-prone sequences |

### Final Dataset

- **Train:** 811,650 sequences (50% pos / 50% neg)
- **Test:** 9,894 sequences (50% pos / 50% neg)
- Gene-level split (no data leakage between train/test)

### Scripts Created

```
scripts/
├── utils.py                    # Shared utilities (hash, validation, paths)
├── extract_positives.py        # Extract wild-types + high-fitness from ProteinGym
├── fetch_uniprot_wildtypes.py  # Pull 10K human sequences from UniProt API
├── ingest_esol.py              # Process ESOL low-yield data
├── ingest_aggregation.py       # Process aggregation-prone data
├── merge_hard_negatives.py     # Combine all negative sources
├── balance_dataset.py          # Create balanced train/test splits
└── inject_wildtype_sequences.py # Inject proteome wild-types
```

---

## Phase 2: Training Infrastructure (Complete)

### Optimizations Implemented

| Issue | Before | After | Impact |
|-------|--------|-------|--------|
| Tokenization | Per-batch in `__getitem__` | Pre-tokenize in `__init__` | **3-4 hours saved** |
| Wild-type imbalance | 160 vs 405K (2500:1) | 10K vs 405K (40:1) + WeightedSampler | Better representation |
| `hash_sequence()` | Duplicated 3x | Shared in utils.py | DRY code |
| iterrows() loops | O(n) Python loops | Vectorized pandas | **100x faster** |
| Memory | Full model in VRAM | Gradient checkpointing | Fits on T4/A100 |

### Training Configuration

- **Model:** ESM-2 650M (`facebook/esm2_t33_650M_UR50D`)
- **Max Length:** 1024 tokens
- **Batch Size:** 16 (A100) / 8 (T4)
- **Gradient Accumulation:** 4 steps (effective batch 64)
- **Learning Rate:** 2e-5
- **Epochs:** 3
- **Mixed Precision:** Enabled (AMP)
- **Gradient Checkpointing:** Enabled

### Training Script Features (`train_colab.py`)

- ESM-2 650M backbone with frozen lower layers
- Mixed precision (AMP) training for memory efficiency
- Gradient accumulation for larger effective batch size
- WeightedRandomSampler to balance rare sources (wild-types)
- Early stopping on F1 score
- Saves best model checkpoint to Google Drive

---

## File Locations

### Data Files
```
data/processed/
├── wildtype_positives.csv          # 10,099 wild-type sequences
├── highfitness_positives.csv       # 405,238 high-fitness mutants
├── hard_negatives_merged.csv       # 555,039 negative sequences
├── uniprot_wildtypes.csv           # 9,974 UniProt sequences
└── esol_low_yield.csv              # 2,318 ESOL negatives

data/deathnote_data/
├── train_balanced.csv              # 811,650 sequences (361 MB)
└── test_balanced.csv               # 9,894 sequences (3.1 MB)
```

### For Colab Training
Upload to `/content/drive/MyDrive/DeathNote/`:
- `train_balanced.csv`
- `test_balanced.csv`

---

## Git Commits

| Commit | Description |
|--------|-------------|
| `a9962e7` | Phase 1 data pipeline + Phase 2 training infrastructure |
| `18616c0` | README updates |
| `0a86fc2` | Add huggingface-hub to requirements |

---

## Next Steps

1. **Run Training:** Execute cell 15 in `connect.ipynb` on A100 Colab
2. **Expected Time:** ~2-3 hours for 3 epochs on 811K sequences
3. **Output:** Model saved to `/content/drive/MyDrive/DeathNote/deathnote_model.pt`
4. **Phase 3:** Deploy model to Streamlit app for inference
