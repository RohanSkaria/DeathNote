"""
Inject Wild-Type (Healthy) Human Proteins into ClinVar Training Data
This adds "healthy controls" to teach the model what "normal" looks like.
"""

import pandas as pd
from Bio import SeqIO
import os
import sys

def inject_wildtype_sequences(clinvar_csv_path, fasta_file_path, output_path=None, max_length=1024):
    """
    Inject wild-type sequences from human proteome into ClinVar training data.
    
    Args:
        clinvar_csv_path: Path to existing ClinVar CSV file
        fasta_file_path: Path to human proteome FASTA file
        output_path: Output CSV path (default: adds '_PLUS_WT' to input filename)
        max_length: Maximum sequence length to include (filters out very long sequences)
    
    Returns:
        DataFrame with merged data
    """
    # 1. Load the existing ClinVar data
    print("📂 Loading ClinVar data...")
    if not os.path.exists(clinvar_csv_path):
        raise FileNotFoundError(f"ClinVar CSV not found: {clinvar_csv_path}")
    
    df_clinvar = pd.read_csv(clinvar_csv_path)
    print(f"   Original Data: {len(df_clinvar):,} samples")
    
    # Validate required columns
    if 'full_sequence' not in df_clinvar.columns or 'label' not in df_clinvar.columns:
        raise ValueError("CSV must contain 'full_sequence' and 'label' columns")
    
    # 2. Load the Wild Type Proteome (The "Healthy Controls")
    print("🧬 Loading Human Reference Proteome...")
    if not os.path.exists(fasta_file_path):
        raise FileNotFoundError(f"FASTA file not found: {fasta_file_path}")
    
    wild_type_data = []
    skipped_long = 0
    skipped_short = 0
    
    for record in SeqIO.parse(fasta_file_path, "fasta"):
        # Clean sequence
        seq = str(record.seq).upper()
        
        # Filter out invalid sequences
        if not seq or len(seq.strip()) == 0:
            continue
        
        # Filter sequences that are too long (will be truncated anyway)
        if len(seq) > max_length:
            skipped_long += 1
            continue
        
        # Filter sequences that are too short (not meaningful)
        if len(seq) < 10:
            skipped_short += 1
            continue
        
        # Label = 1 (SAFE) - Wild-type is always safe
        wild_type_data.append({
            'full_sequence': seq, 
            'label': 1  # 100% Safe
        })
    
    df_wt = pd.DataFrame(wild_type_data)
    print(f"   Injected Controls: {len(df_wt):,} samples")
    if skipped_long > 0:
        print(f"   ⚠️  Skipped {skipped_long:,} sequences > {max_length} AA")
    if skipped_short > 0:
        print(f"   ⚠️  Skipped {skipped_short:,} sequences < 10 AA")
    
    # 3. Merge and Save
    print("⚗️  Mixing Data...")
    df_final = pd.concat([df_clinvar, df_wt], ignore_index=True)
    
    # Shuffle
    df_final = df_final.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Generate output path if not provided
    if output_path is None:
        base_name = os.path.splitext(clinvar_csv_path)[0]
        output_path = f"{base_name}_PLUS_WT.csv"
    
    # Save
    df_final.to_csv(output_path, index=False)
    
    print(f"\n🎉 SUCCESS! Created '{output_path}'")
    print(f"📊 New Stats:")
    print(f"   Total Samples: {len(df_final):,}")
    print(f"   Toxic (0): {(df_final['label']==0).sum():,} ({(df_final['label']==0).sum()/len(df_final)*100:.1f}%)")
    print(f"   Safe  (1): {(df_final['label']==1).sum():,} ({(df_final['label']==1).sum()/len(df_final)*100:.1f}%)")
    print(f"   Boost: +{len(df_wt):,} wild-type sequences")
    
    return df_final

if __name__ == '__main__':
    # Default paths for Colab
    clinvar_path = "clinvar_training_data.csv"
    fasta_path = "human_proteome.fasta"
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        clinvar_path = sys.argv[1]
    if len(sys.argv) > 2:
        fasta_path = sys.argv[2]
    
    # Try Colab paths if not found
    if not os.path.exists(clinvar_path):
        colab_paths = [
            f"/content/data/{clinvar_path}",
            f"/content/drive/MyDrive/Death Note/{clinvar_path}",
            f"/content/{clinvar_path}"
        ]
        for path in colab_paths:
            if os.path.exists(path):
                clinvar_path = path
                break
    
    if not os.path.exists(fasta_path):
        colab_paths = [
            f"/content/data/{fasta_path}",
            f"/content/drive/MyDrive/Death Note/{fasta_path}",
            f"/content/{fasta_path}"
        ]
        for path in colab_paths:
            if os.path.exists(path):
                fasta_path = path
                break
    
    try:
        df = inject_wildtype_sequences(clinvar_path, fasta_path)
        print("\n✅ Ready to train with enhanced dataset!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

