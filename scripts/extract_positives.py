#!/usr/bin/env python3
"""
extract_positives.py - Extract positive (safe) samples for classifier training

Sources:
1. Wild-type sequences: Unmutated reference proteins from ProteinGym
2. High-fitness mutants: Mutations with DMS_score > threshold (experimental success)

Output: CSV files ready for merging with hard negatives
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

from utils import (
    deduplicate_by_sequence,
    find_reference_file,
    find_dms_data_folder,
    contains_ambiguous_residues,
    AMBIGUOUS_PATTERN,
    SOURCE_PROTEINGYM_WILDTYPE,
    SOURCE_PROTEINGYM_HIGHFITNESS,
)


def extract_wildtypes(ref_path, min_len=50, max_len=1024):
    """
    Extract wild-type (unmutated) sequences from ProteinGym reference file.
    These are the natural, functional proteins that serve as positive controls.
    """
    print(f"\n   Reading reference file: {ref_path}")
    ref_df = pd.read_csv(ref_path)

    print(f"   Total DMS assays: {len(ref_df)}")

    # Vectorized filtering (much faster than iterrows)
    mask = (
        ref_df['target_seq'].notna() &
        (ref_df['target_seq'].str.len() > 0) &
        (ref_df['target_seq'].str.len() >= min_len) &
        (ref_df['target_seq'].str.len() <= max_len) &
        (~ref_df['target_seq'].str.contains(AMBIGUOUS_PATTERN, regex=True))
    )

    filtered = ref_df[mask].copy()

    # Build output DataFrame vectorized
    result_df = pd.DataFrame({
        'sequence_id': 'WT_' + filtered['DMS_id'],
        'full_sequence': filtered['target_seq'].str.upper(),
        'source': SOURCE_PROTEINGYM_WILDTYPE,
        'failure_type': 'none',
        'label': 1,
        'score': 0.0
    })

    # Deduplicate
    result_df, removed = deduplicate_by_sequence(result_df, 'full_sequence')

    print(f"   Extracted {len(result_df)} unique wild-type sequences ({removed} duplicates removed)")

    return result_df


def extract_high_fitness(dms_path, ref_path, score_threshold=1.0, min_len=50, max_len=1024):
    """
    Extract high-fitness mutants from ProteinGym DMS assays.
    These are mutations that experimentally improved or maintained function.
    """
    print(f"\n   Reading reference file: {ref_path}")
    ref_df = pd.read_csv(ref_path)

    print(f"   DMS data folder: {dms_path}")
    print(f"   Score threshold: > {score_threshold}")

    all_dfs = []
    processed = 0
    total_seqs = 0

    for _, row in ref_df.iterrows():
        dms_id = row['DMS_id']
        filename = row['DMS_filename']
        file_path = os.path.join(dms_path, filename)

        if not os.path.exists(file_path):
            continue

        try:
            df = pd.read_csv(file_path, low_memory=False)
        except Exception as e:
            print(f"   Warning: Could not read {filename}: {e}")
            continue

        if 'DMS_score' not in df.columns or 'mutated_sequence' not in df.columns:
            continue

        # Vectorized filtering
        mask = (
            pd.notna(df['DMS_score']) &
            np.isfinite(df['DMS_score']) &
            (df['DMS_score'] > score_threshold) &
            df['mutated_sequence'].notna() &
            (df['mutated_sequence'].str.len() > 0) &
            (df['mutated_sequence'].str.len() >= min_len) &
            (df['mutated_sequence'].str.len() <= max_len) &
            (~df['mutated_sequence'].str.contains(AMBIGUOUS_PATTERN, regex=True))
        )

        high_fit = df[mask]

        if high_fit.empty:
            continue

        # Build output DataFrame vectorized (no iterrows)
        result = pd.DataFrame({
            'sequence_id': dms_id + '_' + high_fit['mutant'].fillna('unknown').astype(str),
            'full_sequence': high_fit['mutated_sequence'].str.upper(),
            'source': SOURCE_PROTEINGYM_HIGHFITNESS,
            'failure_type': 'none',
            'label': 1,
            'score': high_fit['DMS_score']
        })

        all_dfs.append(result)
        processed += 1
        total_seqs += len(result)

        if processed % 20 == 0:
            print(f"      Processed {processed} assays, {total_seqs} sequences...")

    print(f"   Processed {processed} assays")

    if not all_dfs:
        return None

    result_df = pd.concat(all_dfs, ignore_index=True)

    # Deduplicate
    result_df, removed = deduplicate_by_sequence(result_df, 'full_sequence')

    print(f"   Extracted {len(result_df)} unique high-fitness sequences ({removed} duplicates removed)")

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract positive samples (wild-types + high-fitness mutants)'
    )
    parser.add_argument('--wildtype-output', type=str, default='wildtype_positives.csv',
                       help='Output filename for wild-types')
    parser.add_argument('--highfitness-output', type=str, default='highfitness_positives.csv',
                       help='Output filename for high-fitness mutants')
    parser.add_argument('--score-threshold', type=float, default=1.0,
                       help='DMS score threshold for high-fitness (default: 1.0)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--skip-wildtype', action='store_true',
                       help='Skip wild-type extraction')
    parser.add_argument('--skip-highfitness', action='store_true',
                       help='Skip high-fitness extraction')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)

    # Find paths using shared utilities
    ref_path = find_reference_file(base_dir)
    dms_path = find_dms_data_folder(base_dir)

    if not ref_path:
        print("Error: Could not find DMS_substitutions.csv reference file")
        sys.exit(1)

    print("=" * 60)
    print("Positive Sample Extraction")
    print("=" * 60)

    total_positives = 0

    # Extract wild-types
    if not args.skip_wildtype:
        print("\n[1/2] Extracting wild-type sequences...")
        wt_df = extract_wildtypes(ref_path, args.min_length, args.max_length)

        if wt_df is not None and len(wt_df) > 0:
            wt_path = os.path.join(output_dir, args.wildtype_output)
            wt_df.to_csv(wt_path, index=False)
            print(f"   Saved to: {wt_path}")
            total_positives += len(wt_df)
    else:
        print("\n[1/2] Skipping wild-type extraction...")

    # Extract high-fitness mutants
    if not args.skip_highfitness:
        print("\n[2/2] Extracting high-fitness mutants...")

        if not dms_path:
            print("   Error: Could not find DMS data folder")
        else:
            hf_df = extract_high_fitness(
                dms_path, ref_path,
                score_threshold=args.score_threshold,
                min_len=args.min_length,
                max_len=args.max_length
            )

            if hf_df is not None and len(hf_df) > 0:
                hf_path = os.path.join(output_dir, args.highfitness_output)
                hf_df.to_csv(hf_path, index=False)
                print(f"   Saved to: {hf_path}")
                total_positives += len(hf_df)
    else:
        print("\n[2/2] Skipping high-fitness extraction...")

    print("\n" + "=" * 60)
    print(f" COMPLETE: {total_positives:,} total positive samples extracted")
    print("=" * 60)


if __name__ == "__main__":
    main()
