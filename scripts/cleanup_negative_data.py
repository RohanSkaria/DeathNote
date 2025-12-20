import pandas as pd
import os
import shutil
from datetime import datetime

def cleanup_duplicates(input_file, output_file=None, backup=True):
    """
    Remove duplicate rows from the negative data CSV file.
    Keeps the first occurrence of each unique sequence_id.
    """
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    print(f"📂 Loading dataset from: {input_file}")
    print("   This may take a moment for large files...")
    
    # Load in chunks if file is very large
    try:
        df = pd.read_csv(input_file, low_memory=False)
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    initial_count = len(df)
    initial_unique = df['sequence_id'].nunique()
    duplicates = initial_count - initial_unique
    
    print(f"\n📊 Before cleanup:")
    print(f"   Total rows: {initial_count:,}")
    print(f"   Unique sequence_ids: {initial_unique:,}")
    print(f"   Duplicate rows: {duplicates:,}")
    print(f"   Duplicate percentage: {duplicates/initial_count*100:.2f}%")
    
    if duplicates == 0:
        print("\n✅ No duplicates found! Dataset is already clean.")
        return
    
    # Create backup if requested
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = input_file.replace('.csv', f'_backup_{timestamp}.csv')
        print(f"\n💾 Creating backup: {backup_file}")
        shutil.copy2(input_file, backup_file)
        print(f"   ✅ Backup created")
    
    # Remove duplicates - keep first occurrence
    print(f"\n🧹 Removing duplicates (keeping first occurrence)...")
    df_cleaned = df.drop_duplicates(subset=['sequence_id'], keep='first')
    
    final_count = len(df_cleaned)
    removed = initial_count - final_count
    
    print(f"\n📊 After cleanup:")
    print(f"   Total rows: {final_count:,}")
    print(f"   Rows removed: {removed:,}")
    print(f"   Reduction: {removed/initial_count*100:.2f}%")
    
    # Determine output file
    if output_file is None:
        output_file = input_file
    
    # Save cleaned dataset
    print(f"\n💾 Saving cleaned dataset to: {output_file}")
    df_cleaned.to_csv(output_file, index=False)
    
    # Verify
    print(f"\n✅ Verification:")
    verify_df = pd.read_csv(output_file)
    verify_unique = verify_df['sequence_id'].nunique()
    print(f"   Saved rows: {len(verify_df):,}")
    print(f"   Unique sequence_ids: {verify_unique:,}")
    
    if len(verify_df) == verify_unique:
        print(f"   ✅ All duplicates removed successfully!")
    else:
        print(f"   ⚠️  Warning: Still found {len(verify_df) - verify_unique} duplicates")
    
    print(f"\n🎉 Cleanup complete!")
    print(f"   Original file: {input_file}")
    if backup:
        print(f"   Backup file: {backup_file}")
    print(f"   Cleaned file: {output_file}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Clean duplicate rows from negative data CSV')
    parser.add_argument('--input', type=str, 
                       default='data/processed/negative_data_batch_1.csv',
                       help='Input CSV file path')
    parser.add_argument('--output', type=str, default=None,
                       help='Output CSV file path (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Skip creating backup file')
    
    args = parser.parse_args()
    
    cleanup_duplicates(
        input_file=args.input,
        output_file=args.output,
        backup=not args.no_backup
    )

