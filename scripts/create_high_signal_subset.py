#!/usr/bin/env python3
"""
create_high_signal_subset.py - Create distilled high-signal training subset

Creates ~20K concentrated dataset:
- Negatives: All ESOL + Aggregation + worst ProteinGym failures
- Positives: Matched wild-types from UniProt + ProteinGym
"""

import pandas as pd
import os

def create_high_signal_subset():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    output_dir = os.path.join(base_dir, "data", "deathnote_data")

    print("=" * 60)
    print("Creating High-Signal Subset")
    print("=" * 60)

    # === NEGATIVES (Hard failures) ===
    negatives = []

    # 1. All ESOL low-yield (manufacturing failures)
    esol_path = os.path.join(processed_dir, "esol_low_yield.csv")
    if os.path.exists(esol_path):
        esol = pd.read_csv(esol_path)
        esol['label'] = 0
        negatives.append(esol)
        print(f"  ESOL low-yield: {len(esol):,}")

    # 2. All aggregation-prone
    agg_path = os.path.join(processed_dir, "aggregation_prone.csv")
    if os.path.exists(agg_path):
        agg = pd.read_csv(agg_path)
        agg['label'] = 0
        negatives.append(agg)
        print(f"  Aggregation-prone: {len(agg):,}")

    # 3. Worst ProteinGym failures (lowest fitness scores)
    neg_path = os.path.join(processed_dir, "proteingym_hard_negatives.csv")
    if os.path.exists(neg_path):
        pg_neg = pd.read_csv(neg_path)
        # Take the 5000 with lowest scores (most severe failures)
        if 'score' in pg_neg.columns:
            pg_neg_sorted = pg_neg.nsmallest(5000, 'score')
        else:
            pg_neg_sorted = pg_neg.sample(n=min(5000, len(pg_neg)), random_state=42)
        pg_neg_sorted['label'] = 0
        negatives.append(pg_neg_sorted)
        print(f"  ProteinGym worst failures: {len(pg_neg_sorted):,}")

    neg_df = pd.concat(negatives, ignore_index=True)
    print(f"  Total negatives: {len(neg_df):,}")

    # === POSITIVES (Wild-types) ===
    positives = []

    # 1. All wild-types (UniProt + ProteinGym)
    wt_path = os.path.join(processed_dir, "wildtype_positives.csv")
    if os.path.exists(wt_path):
        wt = pd.read_csv(wt_path)
        wt['label'] = 1
        positives.append(wt)
        print(f"  Wild-types: {len(wt):,}")

    pos_df = pd.concat(positives, ignore_index=True)

    # Balance: match positive count to negative count
    target_pos = len(neg_df)
    if len(pos_df) > target_pos:
        pos_df = pos_df.sample(n=target_pos, random_state=42)
    print(f"  Total positives (matched): {len(pos_df):,}")

    # === COMBINE ===
    combined = pd.concat([pos_df, neg_df], ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

    # Ensure required columns
    required_cols = ['sequence_id', 'full_sequence', 'source', 'label']
    for col in required_cols:
        if col not in combined.columns:
            if col == 'sequence_id':
                combined[col] = [f"seq_{i}" for i in range(len(combined))]
            elif col == 'source':
                combined[col] = 'unknown'

    # Rename for SaProt compatibility
    combined = combined.rename(columns={'full_sequence': 'aa_sequence'})

    # Save
    output_path = os.path.join(output_dir, "train_high_signal_20k.csv")
    combined.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print("HIGH-SIGNAL SUBSET CREATED")
    print("=" * 60)
    print(f"  Total sequences: {len(combined):,}")
    print(f"  Positives: {(combined['label'] == 1).sum():,}")
    print(f"  Negatives: {(combined['label'] == 0).sum():,}")
    print(f"  Saved to: {output_path}")

    return combined

if __name__ == "__main__":
    create_high_signal_subset()
