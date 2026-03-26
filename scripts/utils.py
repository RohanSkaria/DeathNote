"""
Shared utilities for Death Note data processing scripts.
"""

import hashlib
import os
import re

# Constants
AMBIGUOUS_RESIDUES = 'XBZJUO'
AMBIGUOUS_PATTERN = re.compile(r'[XBZJUO]')

# Source identifiers
SOURCE_PROTEINGYM_WILDTYPE = 'proteingym_wildtype'
SOURCE_PROTEINGYM_HIGHFITNESS = 'proteingym_highfitness'
SOURCE_PROTEINGYM_AGGREGATION = 'proteingym_aggregation'
SOURCE_ESOL = 'esol'

# ID prefixes
ID_PREFIX_WILDTYPE = 'WT_'
ID_PREFIX_ESOL = 'ESOL_'


def hash_sequence(seq: str) -> str:
    """Create MD5 hash of sequence for deduplication."""
    return hashlib.md5(seq.encode()).hexdigest()


def contains_ambiguous_residues(seq: str) -> bool:
    """Check if sequence contains ambiguous amino acid codes."""
    return bool(AMBIGUOUS_PATTERN.search(seq))


def is_valid_length(seq: str, min_len: int = 50, max_len: int = 1024) -> bool:
    """Check if sequence length is within valid range."""
    return min_len <= len(seq) <= max_len


def deduplicate_by_sequence(df, seq_column: str = 'full_sequence'):
    """
    Deduplicate DataFrame by sequence hash.
    Returns deduplicated DataFrame and count of removed duplicates.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[seq_column], keep='first')
    after = len(df)
    return df, before - after


def find_reference_file(base_dir: str) -> str | None:
    """Find ProteinGym reference file in common locations."""
    paths = [
        os.path.join(base_dir, '..', 'ProteinGym', 'reference_files', 'DMS_substitutions.csv'),
        os.path.join(base_dir, 'ProteinGym', 'reference_files', 'DMS_substitutions.csv'),
    ]
    return next((p for p in paths if os.path.exists(p)), None)


def find_dms_data_folder(base_dir: str) -> str | None:
    """Find DMS data folder in common locations."""
    paths = [
        os.path.join(base_dir, 'archive-data', 'DMS_ProteinGym_substitutions'),
        os.path.join(base_dir, 'DMS_ProteinGym_substitutions'),
    ]
    return next((p for p in paths if os.path.exists(p)), None)


def extract_protein_family(seq_id: str) -> str:
    """
    Extract protein family identifier from sequence ID.
    Used for gene-level train/test splitting.
    """
    if seq_id.startswith(ID_PREFIX_WILDTYPE):
        parts = seq_id[len(ID_PREFIX_WILDTYPE):].split('_')
    elif seq_id.startswith(ID_PREFIX_ESOL):
        return 'ESOL'  # Group all E. coli together
    else:
        parts = seq_id.split('_')

    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]
