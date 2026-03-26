#!/usr/bin/env python3
"""
fetch_uniprot_wildtypes.py - Pull high-confidence human wild-type sequences from UniProt

Downloads reviewed (Swiss-Prot) human protein sequences to supplement
the 160 ProteinGym wild-types with ~10,000 canonical sequences.

UniProt filters:
- organism_id:9606 (Homo sapiens)
- reviewed:true (Swiss-Prot = manually curated, high confidence)
- length:[50 TO 1024] (matches our training data range)
"""

import requests
import pandas as pd
import time
import argparse
import os
import sys

# Add parent directory to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.utils import contains_ambiguous_residues, hash_sequence

UNIPROT_STREAM = "https://rest.uniprot.org/uniprotkb/stream"

def fetch_uniprot_sequences(
    organism_id: int = 9606,  # Human
    reviewed: bool = True,    # Swiss-Prot only
    min_length: int = 50,
    max_length: int = 1024,
    limit: int = 10000,
    batch_size: int = 200  # unused but kept for interface compatibility
) -> list[dict]:
    """
    Fetch sequences from UniProt stream API (more reliable for bulk downloads).

    Returns list of dicts with: accession, entry_name, sequence, protein_name
    """

    # Build query
    query_parts = [
        f"organism_id:{organism_id}",
        f"length:[{min_length} TO {max_length}]"
    ]
    if reviewed:
        query_parts.append("reviewed:true")

    query = " AND ".join(query_parts)

    print(f"Query: {query}")
    print(f"Fetching up to {limit:,} sequences via stream API...")

    params = {
        "query": query,
        "format": "tsv",
        "fields": "accession,id,sequence,protein_name",
        "size": limit
    }

    sequences = []

    try:
        response = requests.get(UNIPROT_STREAM, params=params, timeout=300, stream=True)
        response.raise_for_status()

        lines = response.iter_lines(decode_unicode=True)
        header = next(lines)  # Skip header

        for line in lines:
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            accession, entry_name, sequence, protein_name = parts[0], parts[1], parts[2], parts[3]

            if not sequence:
                continue

            sequences.append({
                "accession": accession,
                "entry_name": entry_name,
                "sequence": sequence,
                "protein_name": protein_name
            })

            if len(sequences) % 1000 == 0:
                print(f"   Fetched {len(sequences):,} sequences...")

            if len(sequences) >= limit:
                break

    except requests.RequestException as e:
        print(f"Error fetching from UniProt: {e}")

    print(f"   Total fetched: {len(sequences):,} sequences")
    return sequences[:limit]


def process_sequences(sequences: list[dict]) -> pd.DataFrame:
    """
    Process raw UniProt sequences into training format.

    Filters out:
    - Sequences with ambiguous residues (X, B, Z, J, U, O)
    - Duplicate sequences
    """

    print(f"\nProcessing {len(sequences):,} sequences...")

    records = []
    seen_hashes = set()
    skipped_ambiguous = 0
    skipped_duplicate = 0

    for entry in sequences:
        seq = entry["sequence"].upper()

        # Skip sequences with ambiguous residues
        if contains_ambiguous_residues(seq):
            skipped_ambiguous += 1
            continue

        # Deduplicate by sequence hash
        seq_hash = hash_sequence(seq)
        if seq_hash in seen_hashes:
            skipped_duplicate += 1
            continue
        seen_hashes.add(seq_hash)

        records.append({
            "sequence_id": f"UniProt_{entry['accession']}",
            "full_sequence": seq,
            "source": "uniprot_human_reviewed",
            "failure_type": "none",
            "label": 1,  # Wild-type = positive/safe
            "score": 0.0  # No DMS score for wild-types
        })

    print(f"   Skipped {skipped_ambiguous:,} with ambiguous residues")
    print(f"   Skipped {skipped_duplicate:,} duplicates")
    print(f"   Final: {len(records):,} sequences")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(
        description="Fetch high-confidence human wild-type sequences from UniProt"
    )
    parser.add_argument("--limit", type=int, default=10000,
                       help="Maximum sequences to fetch (default: 10000)")
    parser.add_argument("--min-length", type=int, default=50,
                       help="Minimum sequence length (default: 50)")
    parser.add_argument("--max-length", type=int, default=1024,
                       help="Maximum sequence length (default: 1024)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output CSV path (default: data/processed/uniprot_wildtypes.csv)")
    parser.add_argument("--merge", action="store_true",
                       help="Merge with existing wildtype_positives.csv")
    args = parser.parse_args()

    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")
    os.makedirs(processed_dir, exist_ok=True)

    output_path = args.output or os.path.join(processed_dir, "uniprot_wildtypes.csv")

    print("=" * 60)
    print("UniProt Wild-Type Sequence Fetcher")
    print("=" * 60)
    print(f"Target: {args.limit:,} human reviewed sequences")
    print(f"Length range: {args.min_length}-{args.max_length} AA")
    print()

    # Fetch from UniProt
    sequences = fetch_uniprot_sequences(
        limit=args.limit,
        min_length=args.min_length,
        max_length=args.max_length
    )

    if not sequences:
        print("Error: No sequences fetched!")
        sys.exit(1)

    # Process into DataFrame
    df = process_sequences(sequences)

    # Save
    df.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")

    # Optionally merge with existing wild-types
    if args.merge:
        existing_path = os.path.join(processed_dir, "wildtype_positives.csv")
        if os.path.exists(existing_path):
            print(f"\nMerging with existing {existing_path}...")
            existing_df = pd.read_csv(existing_path)

            # Deduplicate by sequence
            combined = pd.concat([existing_df, df], ignore_index=True)
            before = len(combined)
            combined = combined.drop_duplicates(subset=["full_sequence"], keep="first")
            after = len(combined)

            merged_path = os.path.join(processed_dir, "wildtype_positives_merged.csv")
            combined.to_csv(merged_path, index=False)

            print(f"   Original ProteinGym wild-types: {len(existing_df):,}")
            print(f"   New UniProt wild-types: {len(df):,}")
            print(f"   Duplicates removed: {before - after:,}")
            print(f"   Total merged: {len(combined):,}")
            print(f"   Saved to: {merged_path}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Sequences fetched: {len(df):,}")
    print(f"Average length: {df['full_sequence'].str.len().mean():.0f} AA")
    print(f"Length range: {df['full_sequence'].str.len().min()}-{df['full_sequence'].str.len().max()} AA")

    return df


if __name__ == "__main__":
    main()
