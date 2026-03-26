#!/usr/bin/env python3
"""
merge_hard_negatives.py - Merge all hard negative datasets into unified training set

Combines:
- ProteinGym fitness failures
- eSOL expression failures
- Aggregation-prone sequences

Performs deduplication by sequence hash and prepares for train/test split.
"""

import pandas as pd
import numpy as np
import hashlib
import argparse
import os
import sys


def hash_sequence(seq):
    """Create MD5 hash of sequence for deduplication."""
    return hashlib.md5(seq.encode()).hexdigest()


def load_and_standardize(filepath, source_name):
    """Load a CSV and standardize columns."""
    if not os.path.exists(filepath):
        print(f"   Warning: {filepath} not found, skipping")
        return None

    df = pd.read_csv(filepath)
    print(f"   {source_name}: {len(df)} rows")

    # Ensure required columns exist
    required = ['sequence_id', 'full_sequence', 'label']
    for col in required:
        if col not in df.columns:
            print(f"   Error: {source_name} missing required column '{col}'")
            return None

    # Standardize columns
    result = pd.DataFrame({
        'sequence_id': df['sequence_id'],
        'full_sequence': df['full_sequence'],
        'source': df['source'] if 'source' in df.columns else source_name,
        'failure_type': df['failure_type'] if 'failure_type' in df.columns else 'unknown',
        'label': df['label']
    })

    # Add score column if available (use different names from different sources)
    if 'dms_score' in df.columns:
        result['score'] = df['dms_score']
    elif 'yield_um' in df.columns:
        result['score'] = df['yield_um']
    else:
        result['score'] = np.nan

    return result


def main():
    parser = argparse.ArgumentParser(
        description='Merge all hard negative datasets'
    )
    parser.add_argument('--output', '-o', type=str,
                       default='hard_negatives_merged.csv',
                       help='Output CSV filename')
    parser.add_argument('--dedupe', action='store_true', default=True,
                       help='Deduplicate by sequence hash (default: True)')
    parser.add_argument('--no-dedupe', dest='dedupe', action='store_false',
                       help='Skip deduplication')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    output_path = os.path.join(processed_dir, args.output)

    print("=" * 60)
    print("Hard Negatives Merge Pipeline")
    print("=" * 60)

    # Define source files
    sources = [
        ('proteingym_hard_negatives.csv', 'proteingym'),
        ('esol_low_yield.csv', 'esol'),
        ('aggregation_prone.csv', 'aggregation'),
    ]

    # Load all sources
    print("\n[1/4] Loading datasets...")
    all_dfs = []

    for filename, source_name in sources:
        filepath = os.path.join(processed_dir, filename)
        df = load_and_standardize(filepath, source_name)
        if df is not None:
            all_dfs.append(df)

    if not all_dfs:
        print("\nError: No datasets found!")
        sys.exit(1)

    # Combine
    print("\n[2/4] Combining datasets...")
    merged_df = pd.concat(all_dfs, ignore_index=True)
    print(f"   Combined total: {len(merged_df)} rows")

    # Clean sequences
    print("\n[3/4] Cleaning sequences...")
    initial_count = len(merged_df)

    # Remove invalid sequences
    merged_df = merged_df[merged_df['full_sequence'].notna()]
    merged_df = merged_df[merged_df['full_sequence'].str.len() > 0]

    # Length filter
    seq_lens = merged_df['full_sequence'].str.len()
    merged_df = merged_df[(seq_lens >= args.min_length) & (seq_lens <= args.max_length)]

    # Remove ambiguous residues
    merged_df = merged_df[~merged_df['full_sequence'].str.contains(r'[XBZJUO]', regex=True, na=False)]

    print(f"   After cleaning: {len(merged_df)} rows (removed {initial_count - len(merged_df)})")

    # Deduplication
    if args.dedupe:
        print("\n[4/4] Deduplicating by sequence hash...")
        before_dedupe = len(merged_df)
        merged_df['seq_hash'] = merged_df['full_sequence'].apply(hash_sequence)
        merged_df = merged_df.drop_duplicates(subset='seq_hash', keep='first')
        merged_df = merged_df.drop(columns=['seq_hash'])
        print(f"   After deduplication: {len(merged_df)} rows (removed {before_dedupe - len(merged_df)} duplicates)")
    else:
        print("\n[4/4] Skipping deduplication...")

    # Save
    merged_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Merged {len(merged_df)} unique hard negatives")
    print(f" Saved to: {output_path}")
    print("=" * 60)

    # Statistics
    print(f"\nStatistics:")
    print(f"  - Total sequences: {len(merged_df):,}")

    # By source
    print(f"\n  By source:")
    for source, count in merged_df['source'].value_counts().items():
        pct = count / len(merged_df) * 100
        print(f"    {source}: {count:,} ({pct:.1f}%)")

    # By failure type
    print(f"\n  By failure type:")
    for ftype, count in merged_df['failure_type'].value_counts().items():
        pct = count / len(merged_df) * 100
        print(f"    {ftype}: {count:,} ({pct:.1f}%)")

    # Length distribution
    seq_lens = merged_df['full_sequence'].str.len()
    print(f"\n  Sequence lengths:")
    print(f"    Min: {seq_lens.min()}")
    print(f"    Max: {seq_lens.max()}")
    print(f"    Mean: {seq_lens.mean():.1f}")
    print(f"    Median: {seq_lens.median():.1f}")


if __name__ == "__main__":
    main()
