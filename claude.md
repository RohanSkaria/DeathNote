# ProteinGym Data Ingestion Pipeline - Analysis & Implementation Guide

 # Comments about Review
 most of the feedback and responses from from Gemini, it is in thinking mode and serving as a
  background auditer of implmenetations, if its prepended by a quesiton its from gemini and can be treated
  with more authority or reliability  

## Repository Structure Analysis

### ProteinGym Repository Layout
```
ProteinGym/
├── reference_files/
│   ├── DMS_substitutions.csv          # Reference file with metadata for all DMS assays
│   ├── DMS_indels.csv                 # Reference file for indel assays
│   ├── clinical_substitutions.csv     # Clinical variants reference
│   └── reference_files_description.md # Column descriptions
├── proteingym/
│   ├── utils/
│   │   ├── data_utils.py              # Contains DMS_file_cleanup function
│   │   └── scoring_utils.py           # Contains get_mutated_sequence() helper
│   └── baselines/                     # Various model implementations
└── scripts/
    └── zero_shot_config.sh            # Configuration with default paths
```

### Key File Locations

1. **Reference File**: `/Users/rohan/Development/DeathNote/ProteinGym/reference_files/DMS_substitutions.csv`
   - Contains metadata for all 217 DMS substitution assays
   - Key columns: `DMS_id`, `DMS_filename`, `target_seq`, `seq_len`

2. **DMS Data Files**: Located at `/Users/rohan/Development/DeathNote/DeathNote/DMS_ProteinGym_substitutions/`
   - Each CSV file corresponds to one DMS assay
   - Filename matches `DMS_filename` column in reference file
   - Files already contain `mutated_sequence` column (no need to compute)

3. **DMS File Structure**: Each DMS CSV contains:
   - `mutant` (str): Mutation code like "A24T" or "A24T:D25N" (multiple mutations)
   - `mutated_sequence` (str): Full amino acid sequence after mutation (already present!)
   - `DMS_score` (float): Fitness score (higher = better fitness)
   - `DMS_score_bin` (int): Binary label (1 = fit, 0 = not fit)

## Data Format Understanding

### Mutation Format
- **Single mutation**: `A24T` = Replace Alanine at position 24 with Threonine
- **Multiple mutations**: `A24T:D25N` = Multiple substitutions separated by `:`
- **Indexing**: Positions are **1-based** by default (position 1 = first amino acid)

### Reference File Columns (Key Ones)
- `DMS_id`: Unique identifier (e.g., "A0A140D2T1_ZIKV_Sourisseau_2019")
- `DMS_filename`: CSV filename (e.g., "A0A140D2T1_ZIKV_Sourisseau_2019.csv")
- `target_seq`: Wild-type (reference) sequence
- `seq_len`: Length of target sequence
- `DMS_total_number_mutants`: Total mutants in the assay
- `raw_mut_offset`: Offset for mutation positions (usually 0.1, meaning positions start at 1)

### DMS Score Interpretation
- **Higher DMS_score = Better fitness**
- Negative scores indicate poor fitness
- 25th percentile cutoff will capture the "worst performing" mutants (failures)

## Implementation: `ingest_proteingym.py`

### Key Features

1. **Automatic Path Detection**: 
   - Checks multiple common locations for DMS data folder
   - Supports both relative and absolute paths
   - Falls back to cache directory if needed

2. **Robust Error Handling**:
   - Validates file existence before processing
   - Handles missing columns gracefully
   - Filters invalid sequences and scores
   - Provides informative error messages

3. **Sequence Validation**:
   - Checks sequence length matches wild-type
   - Validates mutation format
   - Drops invalid entries with warnings

4. **Flexible Configuration**:
   - Command-line arguments for all paths
   - Configurable number of files to process
   - Custom output directory

### Usage

```bash
# Basic usage (auto-detects paths)
python ingest_proteingym.py

# Specify custom paths
python ingest_proteingym.py \
    --proteingym_repo ../ProteinGym \
    --dms_data_folder ./DMS_ProteinGym_substitutions \
    --deathnote_repo . \
    --output_dir data/processed \
    --limit 5
```

### Output Format

The script generates `data/processed/negative_data_batch_1.csv` with columns:
- `sequence_id`: Unique identifier (e.g., "A0A140D2T1_ZIKV_Sourisseau_2019_A24T")
- `full_sequence`: Complete mutated amino acid sequence
- `dms_score`: Original DMS_score value (renamed from DMS_score)
- `label`: Always 0 (negative/failure)

## Key Improvements Made

1. **Path Handling**: Fixed to check DeathNote repo location for DMS data
2. **Sequence Handling**: Uses existing `mutated_sequence` column when available
3. **Error Handling**: More robust validation and informative error messages
4. **Validation**: Added length checks and mutation format validation
5. **Output**: Better summary statistics and progress reporting

## Important Notes

1. **Data Already Processed**: The DMS CSV files already contain `mutated_sequence` column, so we don't need to compute it from `mutant` codes (though the function exists as fallback)

2. **Mutation Position Handling**: 
   - Most assays use 1-based indexing
   - Some may have `raw_mut_offset` adjustments (check reference file)
   - The `mutated_sequence` column is already validated in the source data

3. **Multiple Mutations**: 
   - Some assays include multiple mutants (e.g., "A24T:D25N")
   - The helper function handles this automatically if needed

4. **File Naming**: 
   - DMS filenames match exactly the `DMS_filename` column
   - Case-sensitive

5. **Error Handling**:
   - Script continues processing even if some files fail
   - Provides detailed warnings for skipped files
   - Validates data quality at each step

## Next Steps

1. ✅ Verify DMS data folder location
2. ✅ Implement `ingest_proteingym.py` script
3. ✅ Test on first 5 files
4. 🔄 Expand to process all files if successful
5. 🔄 Add data quality checks and statistics

