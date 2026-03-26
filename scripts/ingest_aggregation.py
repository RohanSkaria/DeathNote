#!/usr/bin/env python3
"""
ingest_aggregation.py - Extract aggregation-prone sequences from ProteinGym

Instead of relying on external databases (WALTZ-DB, AmyPro) which are often
unavailable, we extract aggregation/stability data directly from ProteinGym
DMS assays that specifically measure:
- Amyloid aggregation
- Protein stability/folding
- Solubility

Source: ProteinGym reference_files/DMS_substitutions.csv
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

# Keywords to identify aggregation/stability-related assays
AGGREGATION_KEYWORDS = [
    'aggregat',
    'amyloid',
    'fibril',
    'stability',
    'folding',
    'solubil',
]

# Specific DMS assays known to measure aggregation
AGGREGATION_ASSAYS = [
    'A4_HUMAN_Seuma_2022',  # Amyloid beta aggregation atlas
]


def find_aggregation_assays(reference_df, include_stability=True):
    """
    Find DMS assays related to aggregation from reference file.

    Args:
        reference_df: DataFrame from DMS_substitutions.csv
        include_stability: Also include stability assays (broader coverage)
    """
    aggregation_ids = []

    for idx, row in reference_df.iterrows():
        dms_id = row['DMS_id']

        # Check explicit list
        if dms_id in AGGREGATION_ASSAYS:
            aggregation_ids.append(dms_id)
            continue

        # Search in relevant columns
        search_cols = ['DMS_phenotype_name', 'DMS_phenotype_synonyms',
                       'UniProt_functional_annotation']

        for col in search_cols:
            if col in row.index and pd.notna(row[col]):
                text = str(row[col]).lower()
                for keyword in AGGREGATION_KEYWORDS:
                    if keyword in text:
                        # Skip stability unless explicitly requested
                        if keyword == 'stability' and not include_stability:
                            continue
                        aggregation_ids.append(dms_id)
                        break
                if dms_id in aggregation_ids:
                    break

    return list(set(aggregation_ids))


def process_aggregation_file(file_path, target_seq, dms_id, score_threshold=0,
                             min_len=50, max_len=1024, aggregation_type='amyloid'):
    """
    Process a DMS file to extract aggregation-prone sequences.

    For amyloid assays, HIGHER nucleation scores = MORE aggregation-prone (bad)
    For stability assays, LOWER scores = LESS stable (bad)
    """
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"      Error reading {file_path}: {e}")
        return None

    if 'DMS_score' not in df.columns:
        return None

    # Clean data
    df = df[pd.notna(df['DMS_score']) & np.isfinite(df['DMS_score'])].copy()

    if df.empty:
        return None

    # For amyloid aggregation: high scores = aggregation-prone (hard negatives)
    # For stability: low scores = unstable (hard negatives) - handled by main script

    # Filter for aggregation-prone (high nucleation/aggregation score)
    if aggregation_type == 'amyloid':
        # High scores indicate aggregation propensity
        agg_prone = df[df['DMS_score'] > score_threshold].copy()
    else:
        # For stability, low scores indicate instability
        agg_prone = df[df['DMS_score'] < score_threshold].copy()

    if agg_prone.empty:
        return None

    # Get sequences
    if 'mutated_sequence' in agg_prone.columns:
        agg_prone['full_sequence'] = agg_prone['mutated_sequence']
    else:
        return None

    # Clean
    agg_prone = agg_prone[agg_prone['full_sequence'].notna()]
    agg_prone = agg_prone[agg_prone['full_sequence'].str.len() > 0]

    if agg_prone.empty:
        return None

    # Length filter
    seq_lens = agg_prone['full_sequence'].str.len()
    agg_prone = agg_prone[(seq_lens >= min_len) & (seq_lens <= max_len)]

    if agg_prone.empty:
        return None

    # Filter ambiguous residues
    agg_prone = agg_prone[~agg_prone['full_sequence'].str.contains(r'[XBZJUO]', regex=True)]

    if agg_prone.empty:
        return None

    # Build output
    agg_prone['sequence_id'] = dms_id + "_" + agg_prone['mutant'].astype(str)
    agg_prone['dms_score'] = agg_prone['DMS_score']
    agg_prone['source'] = 'proteingym_aggregation'
    agg_prone['failure_type'] = 'aggregation_prone'
    agg_prone['label'] = 0

    return agg_prone[['sequence_id', 'full_sequence', 'dms_score', 'source', 'failure_type', 'label']]


def main():
    parser = argparse.ArgumentParser(
        description='Extract aggregation-prone sequences from ProteinGym'
    )
    parser.add_argument('--output', '-o', type=str,
                       default='aggregation_prone.csv',
                       help='Output CSV filename')
    parser.add_argument('--score-threshold', type=float, default=0,
                       help='Score threshold for aggregation (default: 0, higher = more prone)')
    parser.add_argument('--include-stability', action='store_true',
                       help='Include stability assays (not just aggregation)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--list-assays', action='store_true',
                       help='Just list found aggregation assays and exit')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Find reference file
    ref_paths = [
        os.path.join(base_dir, '..', 'ProteinGym', 'reference_files', 'DMS_substitutions.csv'),
        os.path.join(base_dir, 'ProteinGym', 'reference_files', 'DMS_substitutions.csv'),
    ]

    ref_path = None
    for p in ref_paths:
        if os.path.exists(p):
            ref_path = p
            break

    if not ref_path:
        print("Error: Could not find DMS_substitutions.csv reference file")
        sys.exit(1)

    print("=" * 60)
    print("Aggregation-Prone Sequence Extraction")
    print("=" * 60)

    # Load reference
    print(f"\n[1/3] Loading reference file...")
    ref_df = pd.read_csv(ref_path)
    print(f"   Total DMS assays: {len(ref_df)}")

    # Find aggregation assays
    print(f"\n[2/3] Finding aggregation-related assays...")
    agg_ids = find_aggregation_assays(ref_df, include_stability=args.include_stability)
    print(f"   Found {len(agg_ids)} aggregation-related assays")

    if args.list_assays:
        print("\nAggregation-related assays:")
        for aid in agg_ids:
            row = ref_df[ref_df['DMS_id'] == aid].iloc[0]
            phenotype = row.get('DMS_phenotype_name', 'N/A')
            print(f"   - {aid}: {phenotype}")
        sys.exit(0)

    # Find DMS data folder
    dms_paths = [
        os.path.join(base_dir, 'archive-data', 'DMS_ProteinGym_substitutions'),
        os.path.join(base_dir, 'DMS_ProteinGym_substitutions'),
    ]

    dms_path = None
    for p in dms_paths:
        if os.path.exists(p):
            dms_path = p
            break

    if not dms_path:
        print("Error: Could not find DMS data folder")
        sys.exit(1)

    print(f"   DMS data folder: {dms_path}")

    # Process assays
    print(f"\n[3/3] Extracting aggregation-prone sequences...")
    all_samples = []
    processed = 0

    for dms_id in agg_ids:
        row = ref_df[ref_df['DMS_id'] == dms_id]
        if row.empty:
            continue

        row = row.iloc[0]
        filename = row['DMS_filename']
        target_seq = row['target_seq']

        file_path = os.path.join(dms_path, filename)
        if not os.path.exists(file_path):
            print(f"   Skipping {dms_id}: file not found")
            continue

        # Determine aggregation type
        phenotype = str(row.get('DMS_phenotype_name', '')).lower()
        if 'amyloid' in phenotype or 'aggregat' in phenotype:
            agg_type = 'amyloid'
        else:
            agg_type = 'stability'

        result = process_aggregation_file(
            file_path, target_seq, dms_id,
            score_threshold=args.score_threshold,
            min_len=args.min_length,
            max_len=args.max_length,
            aggregation_type=agg_type
        )

        if result is not None and len(result) > 0:
            all_samples.append(result)
            processed += 1
            print(f"   {dms_id}: {len(result)} aggregation-prone sequences ({agg_type})")

    if not all_samples:
        print("\nNo aggregation-prone sequences found.")
        sys.exit(1)

    # Combine and save
    final_df = pd.concat(all_samples, ignore_index=True)

    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, args.output)

    final_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Extracted {len(final_df)} aggregation-prone sequences")
    print(f" From {processed} assays")
    print(f" Saved to: {output_path}")
    print("=" * 60)

    # Stats
    print(f"\nStatistics:")
    print(f"  - Total sequences: {len(final_df)}")
    print(f"  - Score range: {final_df['dms_score'].min():.2f} to {final_df['dms_score'].max():.2f}")
    seq_lens = final_df['full_sequence'].str.len()
    print(f"  - Length range: {seq_lens.min()} - {seq_lens.max()} AA")


if __name__ == "__main__":
    main()
