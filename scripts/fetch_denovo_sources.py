#!/usr/bin/env python3
"""
fetch_denovo_sources.py - Fetch de novo protein designs from known repositories

Automated data collection from:
1. CAMEO hard targets (AlphaFold prediction failures)
2. ProteinGym de novo benchmark
3. GitHub repositories (Chroma, OpenFold samples)

Usage:
  python fetch_denovo_sources.py --cameo --output cameo_hard.csv
  python fetch_denovo_sources.py --list-sources
"""

import argparse
import os
import sys
import json
from pathlib import Path
from typing import List, Optional

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)

try:
    import pandas as pd
except ImportError:
    print("Please install pandas: pip install pandas")
    sys.exit(1)


# === DATA SOURCE CATALOG ===
KNOWN_SOURCES = {
    "proteingym_denovo": {
        "url": "https://marks.hms.harvard.edu/proteingym/",
        "description": "ProteinGym De Novo benchmark - synthetic proteins for model generalization",
        "format": "csv",
        "notes": "Look for 'De Novo' or 'Synthetic' subset in their downloads"
    },
    "cameo": {
        "url": "https://www.cameo3d.org/",
        "description": "Continuous Automated Model EvaluatOn - hard targets where AF struggled",
        "format": "fasta",
        "notes": "Weekly updated. Look for 'Hard' difficulty targets with high pLDDT but wrong structure"
    },
    "baker_lab_rfdiffusion": {
        "url": "https://github.com/RosettaCommons/RFdiffusion",
        "description": "RFdiffusion sample outputs and benchmark designs",
        "format": "pdb/fasta",
        "paper": "Watson et al., Nature 2023",
        "notes": "Check Supplementary Data for thousands of binder designs"
    },
    "chroma": {
        "url": "https://github.com/generatebio/chroma",
        "description": "Generative model samples from Generate Biomedicines",
        "format": "pdb",
        "notes": "Check examples/ folder for generated structures"
    },
    "openfold": {
        "url": "https://github.com/aqlaboratory/openfold",
        "description": "Open-source AF2 implementation with benchmark data",
        "format": "fasta",
        "notes": "Check notebooks/ and tests/ for sample sequences"
    },
    "rosettafold": {
        "url": "https://github.com/RosettaCommons/RoseTTAFold",
        "description": "RoseTTAFold benchmark sequences",
        "format": "fasta",
        "notes": "Example inputs in examples/ directory"
    },
    "esm_atlas": {
        "url": "https://esmatlas.com/",
        "description": "ESM Metagenomic Atlas - predicted structures for 600M+ proteins",
        "format": "json/pdb",
        "notes": "API available. Good for finding sequences with high confidence predictions"
    },
    "pdb_redo": {
        "url": "https://pdb-redo.eu/",
        "description": "Re-refined PDB structures - identifies originally mismodeled proteins",
        "format": "pdb",
        "notes": "Good source of 'decoy' candidates - sequences with incorrect original structures"
    },
    "afdb_clusters": {
        "url": "https://alphafold.ebi.ac.uk/",
        "description": "AlphaFold Database clustered representatives",
        "format": "fasta/json",
        "notes": "Can filter by pLDDT to find confident predictions"
    }
}


def list_sources():
    """Print information about known data sources"""
    print("\n=== KNOWN DE NOVO DATA SOURCES ===\n")
    for name, info in KNOWN_SOURCES.items():
        print(f"{name}")
        print(f"  URL: {info['url']}")
        print(f"  Description: {info['description']}")
        print(f"  Format: {info['format']}")
        if "paper" in info:
            print(f"  Paper: {info['paper']}")
        print(f"  Notes: {info['notes']}")
        print()


def fetch_esm_atlas_random(n_sequences: int = 100, min_plddt: float = 90.0) -> pd.DataFrame:
    """Fetch random high-confidence sequences from ESM Atlas"""
    print(f"Fetching {n_sequences} sequences from ESM Atlas with pLDDT >= {min_plddt}...")

    # ESM Atlas API endpoint
    base_url = "https://api.esmatlas.com/fetchPredictedStructure/"

    # This would require actual API implementation
    # For now, provide instructions
    print("\nESM Atlas requires API access. To fetch data:")
    print("1. Visit https://esmatlas.com/explore")
    print("2. Use filters to select high-confidence predictions")
    print("3. Download FASTA files")
    print("4. Run: python source_denovo_data.py --fasta-dir ./downloads\n")

    return pd.DataFrame()


def fetch_github_fasta(repo_url: str, file_path: str, output_dir: str) -> Optional[str]:
    """Fetch a FASTA file from a GitHub repository"""
    # Convert GitHub URL to raw content URL
    if "github.com" in repo_url:
        raw_url = repo_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
    else:
        raw_url = repo_url

    # Construct full URL
    full_url = f"{raw_url}/{file_path}" if not repo_url.endswith(file_path) else raw_url

    print(f"Fetching: {full_url}")

    try:
        response = requests.get(full_url, timeout=30)
        response.raise_for_status()

        # Save to output directory
        os.makedirs(output_dir, exist_ok=True)
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_dir, filename)

        with open(output_path, "w") as f:
            f.write(response.text)

        print(f"Saved to: {output_path}")
        return output_path

    except requests.RequestException as e:
        print(f"Failed to fetch: {e}")
        return None


def generate_fetch_guide():
    """Generate a step-by-step guide for manual data collection"""
    guide = """
# De Novo Protein Design Data Collection Guide

## Priority 1: Research Paper Supplements (Highest Signal)

### Baker Lab / RFdiffusion Papers
1. Go to the Nature 2023 RFdiffusion paper: https://www.nature.com/articles/s41586-023-06415-8
2. Download Supplementary Data (usually Excel/CSV files)
3. Look for columns: sequence, pLDDT, experimental_result (expression/binding)
4. Filter to sequences in 300-500 AA range

### ProteinGym De Novo
1. Visit https://proteingym.org/
2. Navigate to "Benchmarks" > "De Novo"
3. Download the synthetic sequences subset
4. These are specifically designed for testing model generalization

## Priority 2: CAMEO Hard Targets (Hallucination Candidates)

1. Visit https://www.cameo3d.org/
2. Go to "Targets" > "Protein Structure"
3. Filter by difficulty: "Hard"
4. Look for targets where:
   - Top models predicted with high confidence (pLDDT > 80)
   - BUT the actual structure differed significantly (GDT-TS < 50)
5. Download FASTA files for these sequences

## Priority 3: GitHub Repositories

### RFdiffusion Sample Outputs
```bash
git clone https://github.com/RosettaCommons/RFdiffusion
cd RFdiffusion
# Look in examples/ and benchmark/ directories
find . -name "*.fasta" -o -name "*.fa"
```

### Chroma (Generate Bio)
```bash
git clone https://github.com/generatebio/chroma
cd chroma
# Check examples/ for generated designs
```

### OpenFold
```bash
git clone https://github.com/aqlaboratory/openfold
cd openfold
# Look in notebooks/ for test sequences
```

## Priority 4: Decoy Sets

### PDB-REDO
1. Visit https://pdb-redo.eu/
2. Search for entries with significant re-refinement changes
3. These represent originally mismodeled structures
4. The sequences may have properties that cause modeling failures

### Rosetta Decoy Sets
- Look for published "decoy discrimination" benchmark sets
- These contain intentionally unstable/incorrect structures

## After Collection

Run the data through the sourcing pipeline:
```bash
# Convert FASTA files to CSV
python source_denovo_data.py --fasta-dir ./collected_data --output denovo_test.csv

# Filter to ideal range
python source_denovo_data.py --filter denovo_test.csv --output denovo_filtered.csv

# Run Assassin inference
python assassin_inference.py --file denovo_filtered.csv --output assassin_results.csv
```

## Validation Strategy

For each source, track:
1. Source name/paper
2. Design method (RFdiffusion, ProteinMPNN, etc.)
3. Experimental outcome if known (expressed, soluble, active)
4. pLDDT or confidence score

Then compare Assassin predictions to experimental outcomes to measure:
- True Positive Rate: Flagging designs that actually failed
- False Positive Rate: Incorrectly flagging successful designs
- Calibration: Does toxic_prob correlate with failure rate?
"""
    return guide


def main():
    parser = argparse.ArgumentParser(
        description="Fetch de novo protein designs from known repositories",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--list-sources", action="store_true", help="List known data sources")
    parser.add_argument("--guide", action="store_true", help="Print data collection guide")
    parser.add_argument("--output-dir", "-o", default="./denovo_downloads", help="Output directory")
    parser.add_argument("--fetch-github", help="Fetch FASTA from GitHub repo URL")
    parser.add_argument("--file-path", help="File path within GitHub repo")

    args = parser.parse_args()

    if args.list_sources:
        list_sources()

    elif args.guide:
        print(generate_fetch_guide())

    elif args.fetch_github:
        if not args.file_path:
            print("Error: --file-path required when using --fetch-github")
            sys.exit(1)
        fetch_github_fasta(args.fetch_github, args.file_path, args.output_dir)

    else:
        parser.print_help()
        print("\n" + "=" * 60)
        print("Quick Start:")
        print("  python fetch_denovo_sources.py --list-sources")
        print("  python fetch_denovo_sources.py --guide")
        print("=" * 60)


if __name__ == "__main__":
    main()
