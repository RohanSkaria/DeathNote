import pandas as pd
import numpy as np
import os

def validate_dataset(csv_path, num_samples=20):
    """
    Validates the dataset by showing sample rows and statistics.
    Displays raw data for visual inspection.
    """
    if not os.path.exists(csv_path):
        print(f"❌ File not found: {csv_path}")
        return
    
    print("="*80)
    print("DATASET VALIDATION")
    print("="*80)
    print(f"\n📂 Loading: {csv_path}")
    
    # Load dataset
    df = pd.read_csv(csv_path, low_memory=False)
    
    print(f"✅ Loaded {len(df):,} rows")
    
    # Basic statistics
    print("\n" + "="*80)
    print("BASIC STATISTICS")
    print("="*80)
    print(f"Total rows: {len(df):,}")
    print(f"Unique sequence_ids: {df['sequence_id'].nunique():,}")
    print(f"Duplicates: {len(df) - df['sequence_id'].nunique():,}")
    
    # Label distribution
    print(f"\nLabel Distribution:")
    label_counts = df['label'].value_counts().sort_index()
    for label, count in label_counts.items():
        pct = count / len(df) * 100
        label_name = "Negative/Failure" if label == 0 else "Positive/Success"
        print(f"  Label {label} ({label_name}): {count:,} ({pct:.2f}%)")
    
    # Score statistics
    print(f"\nScore Statistics:")
    print(f"  Min: {df['dms_score'].min():.4f}")
    print(f"  Max: {df['dms_score'].max():.4f}")
    print(f"  Mean: {df['dms_score'].mean():.4f}")
    print(f"  Median: {df['dms_score'].median():.4f}")
    print(f"  25th percentile: {df['dms_score'].quantile(0.25):.4f}")
    print(f"  75th percentile: {df['dms_score'].quantile(0.75):.4f}")
    
    # Sequence length statistics
    df['seq_length'] = df['full_sequence'].str.len()
    print(f"\nSequence Length Statistics:")
    print(f"  Min: {df['seq_length'].min()}")
    print(f"  Max: {df['seq_length'].max()}")
    print(f"  Mean: {df['seq_length'].mean():.1f}")
    print(f"  Median: {df['seq_length'].median():.0f}")
    
    # Sample data - show both labels
    print("\n" + "="*80)
    print(f"SAMPLE DATA (showing {num_samples} random samples)")
    print("="*80)
    
    # Get samples from each label
    samples_per_label = num_samples // 2
    
    # Sample negatives
    negatives = df[df['label'] == 0].sample(min(samples_per_label, len(df[df['label'] == 0])))
    
    # Sample positives
    positives = df[df['label'] == 1].sample(min(samples_per_label, len(df[df['label'] == 1])))
    
    # Combine and shuffle
    samples = pd.concat([negatives, positives]).sample(frac=1).head(num_samples)
    
    # Display raw data
    for idx, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n{'─'*80}")
        print(f"Sample {idx}/{num_samples}")
        print(f"{'─'*80}")
        print(f"sequence_id: {row['sequence_id']}")
        print(f"label: {row['label']} ({'Negative' if row['label'] == 0 else 'Positive'})")
        print(f"dms_score: {row['dms_score']:.6f}")
        print(f"sequence_length: {len(row['full_sequence'])}")
        print(f"\nfull_sequence:")
        print(f"{row['full_sequence']}")
    
    # Show some edge cases
    print("\n" + "="*80)
    print("EDGE CASES")
    print("="*80)
    
    # Highest score
    max_score_row = df.loc[df['dms_score'].idxmax()]
    print(f"\n📈 Highest Score:")
    print(f"  sequence_id: {max_score_row['sequence_id']}")
    print(f"  label: {max_score_row['label']}")
    print(f"  dms_score: {max_score_row['dms_score']:.2f}")
    print(f"  sequence_length: {len(max_score_row['full_sequence'])}")
    print(f"  sequence_preview: {max_score_row['full_sequence'][:100]}...")
    
    # Lowest score
    min_score_row = df.loc[df['dms_score'].idxmin()]
    print(f"\n📉 Lowest Score:")
    print(f"  sequence_id: {min_score_row['sequence_id']}")
    print(f"  label: {min_score_row['label']}")
    print(f"  dms_score: {min_score_row['dms_score']:.2f}")
    print(f"  sequence_length: {len(min_score_row['full_sequence'])}")
    print(f"  sequence_preview: {min_score_row['full_sequence'][:100]}...")
    
    # Shortest sequence
    shortest = df.loc[df['seq_length'].idxmin()]
    print(f"\n📏 Shortest Sequence:")
    print(f"  sequence_id: {shortest['sequence_id']}")
    print(f"  label: {shortest['label']}")
    print(f"  dms_score: {shortest['dms_score']:.6f}")
    print(f"  sequence_length: {len(shortest['full_sequence'])}")
    print(f"  sequence: {shortest['full_sequence']}")
    
    # Longest sequence
    longest = df.loc[df['seq_length'].idxmax()]
    print(f"\n📏 Longest Sequence:")
    print(f"  sequence_id: {longest['sequence_id']}")
    print(f"  label: {longest['label']}")
    print(f"  dms_score: {longest['dms_score']:.6f}")
    print(f"  sequence_length: {len(longest['full_sequence'])}")
    print(f"  sequence_preview: {longest['full_sequence'][:100]}...")
    
    # Check for any issues
    print("\n" + "="*80)
    print("VALIDATION CHECKS")
    print("="*80)
    
    issues = []
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        issues.append(f"⚠️  Missing values found:\n{missing[missing > 0]}")
    else:
        print("✅ No missing values")
    
    # Check for duplicate sequence_ids
    duplicates = df['sequence_id'].duplicated().sum()
    if duplicates > 0:
        issues.append(f"⚠️  Found {duplicates} duplicate sequence_ids")
    else:
        print("✅ No duplicate sequence_ids")
    
    # Check for empty sequences
    empty_seqs = (df['full_sequence'].str.len() == 0).sum()
    if empty_seqs > 0:
        issues.append(f"⚠️  Found {empty_seqs} empty sequences")
    else:
        print("✅ No empty sequences")
    
    # Check label values
    invalid_labels = df[~df['label'].isin([0, 1])]
    if len(invalid_labels) > 0:
        issues.append(f"⚠️  Found {len(invalid_labels)} rows with invalid labels (not 0 or 1)")
    else:
        print("✅ All labels are valid (0 or 1)")
    
    # Check for sequences with invalid amino acids
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    invalid_aa_rows = df[~df['full_sequence'].str.replace(' ', '').str.replace('\n', '').apply(
        lambda s: all(c in valid_aas for c in s.upper())
    )]
    if len(invalid_aa_rows) > 0:
        issues.append(f"⚠️  Found {len(invalid_aa_rows)} sequences with invalid amino acids")
        print(f"   Sample invalid sequences:")
        for _, row in invalid_aa_rows.head(3).iterrows():
            print(f"     {row['sequence_id']}: {row['full_sequence'][:50]}...")
    else:
        print("✅ All sequences contain valid amino acids")
    
    # Print issues if any
    if issues:
        print("\n⚠️  ISSUES FOUND:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("\n✅ All validation checks passed!")
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate protein dataset CSV')
    parser.add_argument('--file', type=str, 
                       default='data/processed/protein_data_batch_1.csv',
                       help='Path to CSV file to validate')
    parser.add_argument('--samples', type=int, default=20,
                       help='Number of sample rows to display')
    
    args = parser.parse_args()
    
    validate_dataset(args.file, args.samples)

