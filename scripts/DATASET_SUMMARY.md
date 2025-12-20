# ProteinGym Dataset Ingestion Summary

## Final Dataset: `protein_data_batch_1.csv`

### Dataset Statistics
- **Total Samples**: 2,465,767
- **Unique Sequences**: 2,465,767 (0 duplicates)
- **File Size**: ~1.0 GB

### Label Distribution (Balanced!)
- **Label 0 (Negative/Failure)**: 1,073,455 (43.53%)
- **Label 1 (Positive/Success)**: 1,392,312 (56.47%)

✅ **Dataset is now balanced and ready for training!**

### Score Statistics
- **Min**: -205.50
- **Max**: 100,215,199.65
- **Median**: 0.0105

### Outlier Analysis
- **Rows with score > 100,000**: 170 (0.007%)
- **Affected Datasets**: 3
  1. `B2L11_HUMAN_Dutta_2010_binding-Mcl-1` (93 negatives, 77 positives)
  2. `Q6WV12_9MAXI_Somermeyer_2022` (Activity)
  3. `Q8WTC7_9CNID_Somermeyer_2022` (Activity)

**Note**: These outlier scores are from datasets using raw fluorescence/read counts instead of normalized log-enrichment scores. They are still valid data - the `DMS_score_bin` column correctly identifies them as positive/negative regardless of score magnitude.

### Processing Summary
- **Total DMS Assays Processed**: 217/217 (100% success rate)
- **Method Used**: `DMS_score_bin` column (pre-computed binary labels)
- **Unique DMS Assays**: 4,872 (some assays have multiple variants)

## Key Improvements Made

1. ✅ **Fixed Dataset Balance**: Now extracts both positive (label=1) and negative (label=0) samples
2. ✅ **Used Pre-computed Labels**: Prioritizes `DMS_score_bin` column for reliability
3. ✅ **No Duplicates**: All sequence_ids are unique
4. ✅ **Upsert-Safe**: Script can resume processing if interrupted
5. ✅ **Outlier Handling**: Outlier scores are valid data (raw fluorescence), correctly labeled

## Files Created

- `data/processed/protein_data_batch_1.csv` - **Main balanced dataset** (use this!)
- `data/processed/negative_data_batch_1.csv` - Old negative-only dataset (deprecated)
- `scripts/ingest_proteingym.py` - Updated ingestion script
- `diagnostic_outliers.py` - Diagnostic script for outlier analysis
- `diagnostic_balance.py` - Diagnostic script for balance checking
- `cleanup_negative_data.py` - Cleanup script for removing duplicates

## Usage

### Re-run ingestion (if needed):
```bash
python scripts/ingest_proteingym.py --output_file protein_data_batch_1.csv
```

### Extract negatives only (backward compatibility):
```bash
python scripts/ingest_proteingym.py --negatives-only --output_file negative_only.csv
```

### Check dataset balance:
```bash
python diagnostic_balance.py
```

### Analyze outliers:
```bash
python diagnostic_outliers.py
```

## Next Steps

1. ✅ Dataset is balanced and ready for training
2. ✅ No duplicates to worry about
3. ✅ Outlier scores are correctly labeled (can normalize if needed for model training)
4. Ready to proceed with model training!

