#!/usr/bin/env python3
"""
balance_dataset.py - Merge positives and negatives into balanced train/test sets

Ensures:
1. Gene-level separation between train and test
2. Reasonable class balance (target ~1:1)
3. No sequence duplicates across splits
"""

import pandas as pd
import numpy as np
import argparse
import os
import sys

from utils import deduplicate_by_sequence, extract_protein_family


def load_and_combine(processed_dir):
    """Load all positive and negative datasets."""
    datasets = []

    # Positives
    for filename in ['wildtype_positives.csv', 'highfitness_positives.csv']:
        path = os.path.join(processed_dir, filename)
        if os.path.exists(path):
            df = pd.read_csv(path)
            print(f"   Loaded {len(df):,} from {filename}")
            datasets.append(df)

    # Negatives (merged from Phase 1)
    neg_path = os.path.join(processed_dir, 'hard_negatives_merged.csv')
    if os.path.exists(neg_path):
        df = pd.read_csv(neg_path)
        print(f"   Loaded {len(df):,} from hard_negatives_merged.csv")
        datasets.append(df)

    if not datasets:
        return None

    combined = pd.concat(datasets, ignore_index=True)
    return combined


def gene_level_split(df, train_ratio=0.8, random_state=42):
    """
    Split data ensuring no gene appears in both train and test.
    """
    np.random.seed(random_state)

    # Extract protein families
    df['protein_family'] = df['sequence_id'].apply(extract_protein_family)

    # Get unique families and their sizes
    family_sizes = df.groupby('protein_family').size().to_dict()
    families = list(family_sizes.keys())
    np.random.shuffle(families)

    # Allocate families to train/test
    train_families = set()
    test_families = set()
    train_count = 0
    total = len(df)
    target_train = int(total * train_ratio)

    for family in families:
        if train_count < target_train:
            train_families.add(family)
            train_count += family_sizes[family]
        else:
            test_families.add(family)

    # Split data (drop helper column directly to avoid redundant copy)
    train_df = df[df['protein_family'].isin(train_families)].drop(columns=['protein_family'])
    test_df = df[df['protein_family'].isin(test_families)].drop(columns=['protein_family'])

    return train_df, test_df, len(train_families), len(test_families)


def balance_classes(df, target_ratio=1.0, undersample_majority=True):
    """
    Balance positive and negative classes.

    Args:
        df: DataFrame with 'label' column
        target_ratio: Target neg:pos ratio (1.0 = equal)
        undersample_majority: If True, reduce majority class; else oversample minority
    """
    pos_df = df[df['label'] == 1]
    neg_df = df[df['label'] == 0]

    pos_count = len(pos_df)
    neg_count = len(neg_df)

    print(f"   Before balancing: {pos_count:,} positives, {neg_count:,} negatives")

    if undersample_majority:
        if neg_count > pos_count * target_ratio:
            # Undersample negatives
            target_neg = int(pos_count * target_ratio)
            neg_df = neg_df.sample(n=target_neg, random_state=42)
        elif pos_count > neg_count / target_ratio:
            # Undersample positives
            target_pos = int(neg_count / target_ratio)
            pos_df = pos_df.sample(n=target_pos, random_state=42)

    balanced = pd.concat([pos_df, neg_df], ignore_index=True)
    balanced = balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    new_pos = (balanced['label'] == 1).sum()
    new_neg = (balanced['label'] == 0).sum()
    print(f"   After balancing: {new_pos:,} positives, {new_neg:,} negatives")

    return balanced


def main():
    parser = argparse.ArgumentParser(
        description='Create balanced train/test datasets'
    )
    parser.add_argument('--train-output', type=str, default='train_balanced.csv',
                       help='Output filename for training set')
    parser.add_argument('--test-output', type=str, default='test_balanced.csv',
                       help='Output filename for test set')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                       help='Proportion for training (default: 0.8)')
    parser.add_argument('--balance-ratio', type=float, default=1.0,
                       help='Target neg:pos ratio (default: 1.0 = equal)')
    parser.add_argument('--no-balance', action='store_true',
                       help='Skip class balancing')
    parser.add_argument('--dedupe', action='store_true', default=True,
                       help='Deduplicate by sequence')
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    output_dir = os.path.join(base_dir, 'data', 'deathnote_data')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("Dataset Balancing Pipeline")
    print("=" * 60)

    # Load all data
    print("\n[1/4] Loading datasets...")
    combined = load_and_combine(processed_dir)

    if combined is None or combined.empty:
        print("Error: No data found!")
        sys.exit(1)

    print(f"   Total combined: {len(combined):,}")

    # Deduplicate
    print("\n[2/4] Deduplicating by sequence...")
    combined, removed = deduplicate_by_sequence(combined, 'full_sequence')
    print(f"   Removed {removed:,} duplicates, {len(combined):,} remaining")

    # Gene-level split
    print("\n[3/4] Creating gene-level train/test split...")
    train_df, test_df, train_fams, test_fams = gene_level_split(
        combined, train_ratio=args.train_ratio
    )
    print(f"   Train: {len(train_df):,} sequences from {train_fams} families")
    print(f"   Test: {len(test_df):,} sequences from {test_fams} families")

    # Balance classes
    if not args.no_balance:
        print("\n[4/4] Balancing classes...")
        print("   Training set:")
        train_df = balance_classes(train_df, target_ratio=args.balance_ratio)
        print("   Test set:")
        test_df = balance_classes(test_df, target_ratio=args.balance_ratio)
    else:
        print("\n[4/4] Skipping class balancing...")

    # Save
    train_path = os.path.join(output_dir, args.train_output)
    test_path = os.path.join(output_dir, args.test_output)

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print("\n" + "=" * 60)
    print(" COMPLETE")
    print("=" * 60)
    print(f"\n   Train: {len(train_df):,} → {train_path}")
    print(f"   Test:  {len(test_df):,} → {test_path}")

    # Final statistics
    print("\n   Final Statistics:")
    train_pos = (train_df['label'] == 1).sum()
    train_neg = (train_df['label'] == 0).sum()
    test_pos = (test_df['label'] == 1).sum()
    test_neg = (test_df['label'] == 0).sum()

    print(f"   Train: {train_pos:,} pos ({train_pos/len(train_df)*100:.1f}%), {train_neg:,} neg ({train_neg/len(train_df)*100:.1f}%)")
    print(f"   Test:  {test_pos:,} pos ({test_pos/len(test_df)*100:.1f}%), {test_neg:,} neg ({test_neg/len(test_df)*100:.1f}%)")


if __name__ == "__main__":
    main()
