import pandas as pd
import numpy as np
import os
import argparse
import sys

def apply_mutation(wt_sequence, mutation_code, start_idx=1):
    """
    Fallback helper: Applies a mutation code to a wild-type sequence.
    Only used if 'mutated_sequence' column is missing.
    Returns None on any error to allow filtering invalid entries.
    """
    if pd.isna(mutation_code) or not isinstance(mutation_code, str):
        return None
        
    mutated_seq = list(wt_sequence)
    mutations = mutation_code.split(":")
    
    for mutation in mutations:
        try:
            if len(mutation) < 3:
                return None
                
            from_aa = mutation[0]
            to_aa = mutation[-1]
            pos_str = mutation[1:-1]
            
            if not pos_str.isdigit():
                return None
                
            position = int(pos_str)
            relative_position = position - start_idx
            
            # Bounds check
            if relative_position < 0 or relative_position >= len(wt_sequence):
                return None
            
            # Wild Type mismatch check - CRITICAL: Don't force mutations
            # If wild-type doesn't match, the mutation code is invalid
            if mutated_seq[relative_position] != from_aa:
                return None  # Invalid mutation, return None to filter out
                
            mutated_seq[relative_position] = to_aa
            
        except Exception:
            return None

    return "".join(mutated_seq)

def _process_samples(df_subset, target_seq, dms_id, label):
    """
    Helper function to process a subset of samples (negatives or positives).
    Handles sequence extraction, validation, and standardization.
    """
    if df_subset.empty:
        return None
    
    # Sequence Handling - Prioritize Reliability
    if 'mutated_sequence' in df_subset.columns:
        # Best Case: Use the pre-computed sequence
        df_subset['full_sequence'] = df_subset['mutated_sequence']
    elif 'mutant' in df_subset.columns:
        # Fallback Case: Calculate from mutant column
        df_subset['full_sequence'] = df_subset['mutant'].apply(
            lambda x: apply_mutation(target_seq, x) if pd.notna(x) else None
        )
    else:
        return None

    # Clean up invalid sequences
    initial_count = len(df_subset)
    df_subset = df_subset[df_subset['full_sequence'].notna()]
    df_subset = df_subset[df_subset['full_sequence'].str.len() > 0]
    
    if df_subset.empty:
        return None
    
    # Validation: Length Check
    seq_lens = df_subset['full_sequence'].str.len()
    valid_len = len(target_seq)
    df_subset = df_subset[seq_lens == valid_len]
    
    if df_subset.empty:
        return None
    
    # Standardize Output
    df_subset['label'] = label
    df_subset['sequence_id'] = dms_id + "_" + df_subset['mutant'].astype(str)
    df_subset.rename(columns={'DMS_score': 'dms_score'}, inplace=True)
    
    return df_subset[['sequence_id', 'full_sequence', 'dms_score', 'label']]

def process_dms_file(dms_file_path, target_seq, dms_id, extract_positives=True):
    """
    Reads a single DMS file and extracts both negative and positive samples.
    Prioritizes DMS_score_bin column if available, otherwise uses percentile method.
    """
    try:
        df = pd.read_csv(dms_file_path, low_memory=False)
    except FileNotFoundError:
        print(f"❌ File not found: {dms_file_path}")
        return None
    except Exception as e:
        print(f"❌ Error reading {dms_file_path}: {e}")
        return None

    # Required columns check
    if 'DMS_score' not in df.columns:
        print(f"⚠️  Skipping {dms_id}: 'DMS_score' missing.")
        return None

    # Filter out invalid scores
    df = df[pd.notna(df['DMS_score']) & np.isfinite(df['DMS_score'])].copy()
    
    if df.empty:
        print(f"⚠️  {dms_id}: No valid DMS_score values found.")
        return None

    # Extract samples based on available columns
    if 'DMS_score_bin' in df.columns:
        # BEST: Use pre-computed binary labels
        negatives = df[df['DMS_score_bin'] == 0].copy()
        if extract_positives:
            positives = df[df['DMS_score_bin'] == 1].copy()
        else:
            positives = pd.DataFrame()
        method = "DMS_score_bin"
    else:
        # FALLBACK: Use percentile method
        threshold_25 = df['DMS_score'].quantile(0.25)
        threshold_75 = df['DMS_score'].quantile(0.75)
        negatives = df[df['DMS_score'] < threshold_25].copy()
        if extract_positives:
            positives = df[df['DMS_score'] >= threshold_75].copy()
        else:
            positives = pd.DataFrame()
        method = "percentile"

    if negatives.empty:
        print(f"⚠️  {dms_id}: No negative samples found.")
        return None

    # Process negatives and positives
    all_samples = []
    
    negatives_processed = _process_samples(negatives, target_seq, dms_id, label=0)
    if negatives_processed is not None:
        all_samples.append(negatives_processed)
    
    if extract_positives and not positives.empty:
        positives_processed = _process_samples(positives, target_seq, dms_id, label=1)
        if positives_processed is not None:
            all_samples.append(positives_processed)
    
    if all_samples:
        result = pd.concat(all_samples, ignore_index=True)
        neg_count = (result['label'] == 0).sum()
        pos_count = (result['label'] == 1).sum()
        print(f"   [{method}] Negatives: {neg_count}, Positives: {pos_count}")
        return result
    
    return None

def process_hard_negatives(df, target_seq, dms_id, score_threshold=-1.0, min_len=50, max_len=1024):
    """
    Extract hard negatives: mutations with severe fitness drops (DMS_score < threshold).
    These are high-confidence failures where the model would predict stability.
    """
    # Filter for severe fitness drops
    hard_negs = df[df['DMS_score'] < score_threshold].copy()

    if hard_negs.empty:
        return None

    # Get sequences
    if 'mutated_sequence' in hard_negs.columns:
        hard_negs['full_sequence'] = hard_negs['mutated_sequence']
    elif 'mutant' in hard_negs.columns:
        hard_negs['full_sequence'] = hard_negs['mutant'].apply(
            lambda x: apply_mutation(target_seq, x) if pd.notna(x) else None
        )
    else:
        return None

    # Clean invalid sequences
    hard_negs = hard_negs[hard_negs['full_sequence'].notna()]
    hard_negs = hard_negs[hard_negs['full_sequence'].str.len() > 0]

    if hard_negs.empty:
        return None

    # Length filter
    seq_lens = hard_negs['full_sequence'].str.len()
    hard_negs = hard_negs[(seq_lens >= min_len) & (seq_lens <= max_len)]

    if hard_negs.empty:
        return None

    # Filter out ambiguous residues (X, B, Z, J, U, O)
    ambiguous_pattern = r'[XBZJUO]'
    hard_negs = hard_negs[~hard_negs['full_sequence'].str.contains(ambiguous_pattern, regex=True)]

    if hard_negs.empty:
        return None

    # Standardize output
    hard_negs['sequence_id'] = dms_id + "_" + hard_negs['mutant'].astype(str)
    hard_negs['dms_score'] = hard_negs['DMS_score']
    hard_negs['source'] = 'proteingym'
    hard_negs['failure_type'] = 'fitness_drop'
    hard_negs['label'] = 0

    return hard_negs[['sequence_id', 'full_sequence', 'dms_score', 'source', 'failure_type', 'label']]


def main():
    parser = argparse.ArgumentParser(description='Ingest Positive and Negative Data from ProteinGym')
    parser.add_argument('--limit', type=int, default=None,
                       help='Number of DMS assays to process (None = process all)')
    parser.add_argument('--output_file', type=str, default='protein_data_batch_1.csv',
                       help='Output filename (default: protein_data_batch_1.csv)')
    parser.add_argument('--negatives-only', action='store_true',
                       help='Only extract negative samples (for backward compatibility)')
    parser.add_argument('--hard-negatives', action='store_true',
                       help='Extract hard negatives only (DMS_score < threshold)')
    parser.add_argument('--score-threshold', type=float, default=-1.0,
                       help='DMS score threshold for hard negatives (default: -1.0)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length (default: 50)')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    args = parser.parse_args()
    
    # Auto-detect paths assuming we are inside "DeathNote"
    base_dir = os.getcwd()
    
    # 1. Locate Data Folder (Prioritize local folder)
    possible_data_paths = [
        os.path.join(base_dir, "archive-data", "DMS_ProteinGym_substitutions"),
        os.path.join(base_dir, "DMS_ProteinGym_substitutions"),
        os.path.expanduser("~/.cache/ProteinGym/DMS_ProteinGym_substitutions"),
        os.path.join(base_dir, "..", "ProteinGym", "DMS_ProteinGym_substitutions")
    ]
    
    dms_path = next((p for p in possible_data_paths if os.path.exists(p)), None)
    
    # 2. Locate Reference File
    possible_ref_paths = [
        os.path.join(base_dir, "reference_files", "DMS_substitutions.csv"),
        os.path.join(base_dir, "..", "ProteinGym", "reference_files", "DMS_substitutions.csv")
    ]
    ref_path = next((p for p in possible_ref_paths if os.path.exists(p)), None)

    if not dms_path or not ref_path:
        print("❌ Critical Error: Could not locate Data or Reference files.")
        print(f"   Searched for data in: {possible_data_paths}")
        print(f"   Searched for ref in: {possible_ref_paths}")
        sys.exit(1)

    print(f"✅ Data Source: {dms_path}")
    print(f"✅ Reference:   {ref_path}")

    # Setup output path
    output_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output_file)
    
    # Process reference file first to get DMS_id list
    try:
        ref_df = pd.read_csv(ref_path)
    except Exception as e:
        sys.exit(f"❌ Error reading reference file: {e}")
    
    # Upsert-safe: Load existing data if it exists
    existing_df = None
    processed_assays = set()
    if os.path.exists(output_path):
        try:
            existing_df = pd.read_csv(output_path)
            # Extract processed DMS IDs from sequence_id column
            # Format: DMS_id_mutant, where DMS_id can have varying number of parts
            # Match sequence_ids against known DMS_ids from reference file
            sequence_ids = existing_df['sequence_id'].astype(str).unique()
            known_dms_ids = set(ref_df['DMS_id'].values)
            
            for seq_id in sequence_ids:
                # Find matching DMS_id by checking if sequence_id starts with any known DMS_id
                for dms_id in known_dms_ids:
                    if seq_id.startswith(dms_id + '_'):
                        processed_assays.add(dms_id)
                        break
            
            print(f"📂 Found existing file with {len(existing_df)} rows from {len(processed_assays)} assays")
            print(f"   Will skip already processed assays and append new data")
        except Exception as e:
            print(f"⚠️  Warning: Could not read existing file: {e}")
            print(f"   Starting fresh...")
            existing_df = None
            processed_assays = set()
    
    # Filter to unprocessed assays if resuming
    total_assays = len(ref_df)
    if processed_assays:
        ref_df = ref_df[~ref_df['DMS_id'].isin(processed_assays)]
        skipped_count = total_assays - len(ref_df)
        print(f"📋 {len(ref_df)} assays remaining to process (skipping {skipped_count} already processed)")
    else:
        print(f"📋 Processing all {total_assays} assays")
    
    # Apply limit if specified
    if args.limit:
        targets = ref_df.head(args.limit)
        print(f"\n🚀 Processing {args.limit} assays (limited)...\n")
    else:
        targets = ref_df
        print(f"\n🚀 Processing all {len(targets)} assays...\n")
    
    all_samples = []
    processed_count = 0
    failed_count = 0

    # Print mode info
    if args.hard_negatives:
        print(f"🎯 Mode: HARD NEGATIVES (score < {args.score_threshold}, length {args.min_length}-{args.max_length})")

    for idx, (_, row) in enumerate(targets.iterrows(), 1):
        dms_id = row['DMS_id']
        filename = row['DMS_filename']
        target_seq = row['target_seq']

        full_path = os.path.join(dms_path, filename)

        if args.hard_negatives:
            # Hard negatives mode: stricter filtering
            try:
                df = pd.read_csv(full_path, low_memory=False)
                df = df[pd.notna(df['DMS_score']) & np.isfinite(df['DMS_score'])].copy()
                result = process_hard_negatives(
                    df, target_seq, dms_id,
                    score_threshold=args.score_threshold,
                    min_len=args.min_length,
                    max_len=args.max_length
                )
            except Exception as e:
                print(f"   [{idx}/{len(targets)}] ❌ {dms_id}: Error - {e}")
                failed_count += 1
                continue
        else:
            # Standard mode
            result = process_dms_file(full_path, target_seq, dms_id, extract_positives=not args.negatives_only)

        if result is not None and len(result) > 0:
            all_samples.append(result)
            processed_count += 1
            if args.hard_negatives:
                print(f"   [{idx}/{len(targets)}] ✅ {dms_id}: {len(result)} hard negatives")
            else:
                neg_count = (result['label'] == 0).sum()
                pos_count = (result['label'] == 1).sum()
                print(f"   [{idx}/{len(targets)}] ✅ {dms_id}: {neg_count} neg, {pos_count} pos")
        else:
            failed_count += 1
            print(f"   [{idx}/{len(targets)}] ⚠️  {dms_id}: No samples matched criteria")

    # Combine with existing data (upsert-safe)
    if all_samples:
        new_df = pd.concat(all_samples, ignore_index=True)
        
        # Append to existing data if it exists (no deduplication - keep all rows)
        if existing_df is not None:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"\n📊 Merged: {len(existing_df)} existing + {len(new_df)} new = {len(final_df)} total rows")
        else:
            final_df = new_df
        
        # Save (no deduplication - we keep all rows even if sequence_id duplicates)
        final_df.to_csv(output_path, index=False)
        print("="*60)
        print(f"🎉 SUCCESS: Extracted {len(new_df)} new samples")
        print(f"💾 Saved to: {output_path}")
        print(f"📈 Total rows in file: {len(final_df)}")
        print("="*60)
        print(f"\nSummary:")
        print(f"  - New sequences added: {len(new_df)}")
        print(f"  - Total sequences in file: {len(final_df)}")
        print(f"  - Successfully processed: {processed_count}/{len(targets)}")
        print(f"  - Failed: {failed_count}/{len(targets)}")
        
        # Label distribution
        label_counts = final_df['label'].value_counts().sort_index()
        print(f"\n  Label Distribution:")
        for label, count in label_counts.items():
            print(f"    Label {label}: {count:,} ({count/len(final_df)*100:.2f}%)")
        
        # Count unique DMS assays
        unique_assays = final_df['sequence_id'].str.split('_', n=4).str[:4].str.join('_').nunique()
        print(f"  - Unique DMS assays: {unique_assays}")
        print(f"  - Score range: [{final_df['dms_score'].min():.4f}, {final_df['dms_score'].max():.4f}]")
        print(f"\n⚠️  Note: No deduplication applied - all rows are preserved")
    else:
        if existing_df is not None:
            print(f"\n⚠️  No new data extracted, but existing file preserved at {output_path}")
        else:
            print("\n❌ No data extracted. Check file paths or CSV formats.")

if __name__ == "__main__":
    main()

