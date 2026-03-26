#!/usr/bin/env python3
"""
ingest_esol.py - Extract low-yield proteins from eSOL database

eSOL contains E. coli protein expression data from cell-free synthesis.
We extract proteins with low yield as "hard negatives" - proteins that
fail to express well despite being valid sequences.

Source: https://www.tanpaku.org/tp-esol/
"""

import pandas as pd
import numpy as np
import requests
import argparse
import os
import sys
import hashlib
from io import StringIO

ESOL_DATA_URL = "https://www.tanpaku.org/tp-esol/data/quant_mod3.tab"
ESOL_CACHE_FILE = "esol_raw_data.tab"


def download_esol_data(cache_dir, force_download=False):
    """Download eSOL data file, using cache if available."""
    cache_path = os.path.join(cache_dir, ESOL_CACHE_FILE)

    if os.path.exists(cache_path) and not force_download:
        print(f"   Using cached data: {cache_path}")
        return cache_path

    print(f"   Downloading from {ESOL_DATA_URL}...")
    try:
        response = requests.get(ESOL_DATA_URL, timeout=60)
        response.raise_for_status()

        os.makedirs(cache_dir, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            f.write(response.text)

        print(f"   Saved to {cache_path}")
        return cache_path
    except requests.RequestException as e:
        print(f"   Download failed: {e}")
        return None


def fetch_uniprot_sequence(gene_name, locus_tag):
    """Fetch protein sequence from UniProt for E. coli K-12 gene."""
    # Try multiple query strategies
    queries = [
        f"gene:{gene_name} AND organism_id:83333",  # E. coli K-12
        f"gene:{locus_tag} AND organism_id:83333",
        f"{gene_name} AND organism_id:511145",  # E. coli K-12 MG1655
    ]

    for query in queries:
        try:
            url = f"https://rest.uniprot.org/uniprotkb/search?query={query}&format=fasta&size=1"
            response = requests.get(url, timeout=30)
            if response.status_code == 200 and response.text.strip():
                lines = response.text.strip().split('\n')
                if len(lines) > 1:
                    sequence = ''.join(lines[1:])
                    if len(sequence) >= 20:
                        return sequence
        except Exception:
            continue

    return None


def process_esol_data(data_path, yield_threshold=5.0, min_len=50, max_len=1024,
                      fetch_sequences=False, max_samples=None):
    """
    Process eSOL data to extract low-yield proteins.

    Args:
        data_path: Path to eSOL TSV file
        yield_threshold: Maximum yield in uM to consider as "low yield" (default: 5.0)
        min_len: Minimum sequence length
        max_len: Maximum sequence length
        fetch_sequences: Whether to fetch sequences from UniProt (slow)
        max_samples: Maximum number of samples to process (for testing)
    """
    print(f"\n   Reading eSOL data from {data_path}...")

    try:
        df = pd.read_csv(data_path, sep='\t', low_memory=False)
    except Exception as e:
        print(f"   Error reading file: {e}")
        return None

    print(f"   Total entries: {len(df)}")

    # Standardize column names (handle potential variations)
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if 'yield' in col_lower and 'um' in col_lower:
            col_mapping[col] = 'yield_um'
        elif 'solubility' in col_lower:
            col_mapping[col] = 'solubility'
        elif col_lower == 'gene name k-12' or col_lower == 'gene_name':
            col_mapping[col] = 'gene_name'
        elif col_lower == 'jw_id' or col_lower == 'jw id':
            col_mapping[col] = 'jw_id'
        elif col_lower == 'locustag k-12':
            col_mapping[col] = 'locus_tag'
        elif 'gene product' in col_lower and 'description' in col_lower:
            col_mapping[col] = 'description'
        elif 'mw' in col_lower or 'molecular weight' in col_lower:
            col_mapping[col] = 'mw_kda'

    df = df.rename(columns=col_mapping)

    # Check required columns
    if 'yield_um' not in df.columns:
        print("   Error: Could not find yield column")
        print(f"   Available columns: {list(df.columns)}")
        return None

    # Convert yield to numeric
    df['yield_um'] = pd.to_numeric(df['yield_um'], errors='coerce')

    # Filter for low yield
    initial_count = len(df)
    df = df[df['yield_um'].notna()]
    df = df[df['yield_um'] < yield_threshold]

    print(f"   Low yield (< {yield_threshold} uM): {len(df)} entries")

    if df.empty:
        print("   No low-yield proteins found")
        return None

    # Apply max_samples limit if specified
    if max_samples and len(df) > max_samples:
        df = df.head(max_samples)
        print(f"   Limited to {max_samples} samples for processing")

    # Build output dataframe
    results = []

    if fetch_sequences:
        print(f"\n   Fetching sequences from UniProt (this may take a while)...")
        for idx, row in df.iterrows():
            gene_name = row['gene_name'] if 'gene_name' in row.index else ''
            locus_tag = row['locus_tag'] if 'locus_tag' in row.index else ''
            jw_id = row['jw_id'] if 'jw_id' in row.index else f'ESOL_{idx}'

            # Handle NaN values
            if pd.isna(gene_name) if not isinstance(gene_name, str) else False:
                gene_name = ''
            if pd.isna(locus_tag) if not isinstance(locus_tag, str) else False:
                locus_tag = ''
            if pd.isna(jw_id) if not isinstance(jw_id, str) else False:
                jw_id = f'ESOL_{idx}'

            sequence = fetch_uniprot_sequence(str(gene_name), str(locus_tag))

            if sequence and min_len <= len(sequence) <= max_len:
                # Check for ambiguous residues
                if not any(c in sequence.upper() for c in 'XBZJUO'):
                    seq_id = f"ESOL_{jw_id}_{gene_name}" if gene_name else f"ESOL_{jw_id}"
                    results.append({
                        'sequence_id': seq_id,
                        'full_sequence': sequence.upper(),
                        'yield_um': row['yield_um'],
                        'solubility': row.get('solubility', np.nan),
                        'source': 'esol',
                        'failure_type': 'low_expression',
                        'label': 0
                    })

            if len(results) % 50 == 0 and len(results) > 0:
                print(f"      Processed {len(results)} sequences...")
    else:
        # Without sequence fetching, just prepare metadata
        for idx, row in df.iterrows():
            gene_name = row['gene_name'] if 'gene_name' in row.index else ''
            jw_id = row['jw_id'] if 'jw_id' in row.index else f'ESOL_{idx}'

            # Handle NaN values
            if pd.isna(gene_name) if not isinstance(gene_name, str) else False:
                gene_name = ''
            if pd.isna(jw_id) if not isinstance(jw_id, str) else False:
                jw_id = f'ESOL_{idx}'

            seq_id = f"ESOL_{jw_id}_{gene_name}" if gene_name else f"ESOL_{jw_id}"
            results.append({
                'sequence_id': seq_id,
                'full_sequence': '',  # Will need to be fetched later
                'yield_um': row['yield_um'],
                'solubility': row.get('solubility', np.nan),
                'gene_name': gene_name,
                'source': 'esol',
                'failure_type': 'low_expression',
                'label': 0
            })

    if not results:
        print("   No valid sequences found")
        return None

    result_df = pd.DataFrame(results)
    return result_df


def main():
    parser = argparse.ArgumentParser(
        description='Extract low-yield proteins from eSOL database'
    )
    parser.add_argument('--output', '-o', type=str,
                       default='esol_low_yield.csv',
                       help='Output CSV filename')
    parser.add_argument('--yield-threshold', '-y', type=float, default=5.0,
                       help='Maximum yield in uM to consider as low (default: 5.0)')
    parser.add_argument('--min-length', type=int, default=50,
                       help='Minimum sequence length (default: 50)')
    parser.add_argument('--max-length', type=int, default=1024,
                       help='Maximum sequence length (default: 1024)')
    parser.add_argument('--fetch-sequences', action='store_true',
                       help='Fetch sequences from UniProt (slow, ~1 req/sec)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to process (for testing)')
    parser.add_argument('--force-download', action='store_true',
                       help='Force re-download of eSOL data')
    parser.add_argument('--cache-dir', type=str, default=None,
                       help='Directory for cached downloads')
    args = parser.parse_args()

    # Setup paths
    base_dir = os.getcwd()
    cache_dir = args.cache_dir or os.path.join(base_dir, 'data', 'cache')
    output_dir = os.path.join(base_dir, 'data', 'processed')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    output_path = os.path.join(output_dir, args.output)

    print("=" * 60)
    print("eSOL Low-Yield Protein Extraction")
    print("=" * 60)
    print(f"Yield threshold: < {args.yield_threshold} uM")
    print(f"Sequence length: {args.min_length}-{args.max_length} AA")
    print(f"Fetch sequences: {args.fetch_sequences}")

    # Download data
    print("\n[1/2] Downloading eSOL data...")
    data_path = download_esol_data(cache_dir, args.force_download)

    if not data_path:
        sys.exit(1)

    # Process data
    print("\n[2/2] Processing low-yield proteins...")
    result_df = process_esol_data(
        data_path,
        yield_threshold=args.yield_threshold,
        min_len=args.min_length,
        max_len=args.max_length,
        fetch_sequences=args.fetch_sequences,
        max_samples=args.max_samples
    )

    if result_df is None or result_df.empty:
        print("\n No data extracted.")
        sys.exit(1)

    # Save results
    result_df.to_csv(output_path, index=False)

    print("\n" + "=" * 60)
    print(f" SUCCESS: Extracted {len(result_df)} low-yield proteins")
    print(f" Saved to: {output_path}")
    print("=" * 60)

    # Stats
    print(f"\nStatistics:")
    print(f"  - Total entries: {len(result_df)}")
    print(f"  - Yield range: {result_df['yield_um'].min():.2f} - {result_df['yield_um'].max():.2f} uM")

    if 'solubility' in result_df.columns:
        valid_sol = result_df['solubility'].dropna()
        if len(valid_sol) > 0:
            print(f"  - Solubility range: {valid_sol.min():.1f}% - {valid_sol.max():.1f}%")

    if args.fetch_sequences:
        seq_lens = result_df['full_sequence'].str.len()
        print(f"  - Sequence length range: {seq_lens.min()} - {seq_lens.max()} AA")
    else:
        print("\n  Note: Run with --fetch-sequences to retrieve protein sequences from UniProt")


if __name__ == "__main__":
    main()
