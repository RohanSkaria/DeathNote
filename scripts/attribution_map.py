#!/usr/bin/env python3
"""
attribution_map.py - Identify which residues trigger toxicity predictions

Uses gradient-based saliency to show which amino acids contribute most
to the TOXIC vs SAFE verdict.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import pandas as pd
import numpy as np
import argparse
import os

# === CONFIG ===
BASE_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
DEFAULT_ADAPTER_PATH = os.path.join(project_dir, "saprot_assassin")


def to_saprot(sequence: str) -> str:
    """Convert AA sequence to SaProt format"""
    return "#".join(list(str(sequence).upper().strip())) + "#"


def load_model(adapter_path: str = None):
    """Load model with gradients enabled"""
    adapter_path = adapter_path or DEFAULT_ADAPTER_PATH

    print("Loading SaProt Assassin for attribution...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
        torch_dtype=torch.float32  # Need float32 for gradients
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    # Keep model in train mode for gradient computation
    model.train()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def compute_saliency(sequence: str, tokenizer, model, device, target_class=0):
    """
    Compute input perturbation-based saliency for each residue.

    Uses leave-one-out approach: measure prediction change when each residue is masked.
    target_class: 0 = toxic, 1 = safe
    Returns: list of (residue, saliency_score) tuples
    """
    saprot_seq = to_saprot(sequence)

    # Get baseline prediction
    inputs = tokenizer(
        saprot_seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    model.eval()
    with torch.no_grad():
        baseline_logits = model(**inputs).logits
        baseline_prob = F.softmax(baseline_logits, dim=-1)[0, target_class].item()

    # Compute importance of each residue by masking
    saliency_scores = []

    for i in range(len(sequence)):
        # Create masked sequence (replace residue with X)
        masked_seq = sequence[:i] + 'X' + sequence[i+1:]
        masked_saprot = to_saprot(masked_seq)

        inputs = tokenizer(
            masked_saprot,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            masked_logits = model(**inputs).logits
            masked_prob = F.softmax(masked_logits, dim=-1)[0, target_class].item()

        # Saliency = how much the prediction drops when this residue is masked
        importance = baseline_prob - masked_prob
        saliency_scores.append((sequence[i], abs(importance), i))

    return saliency_scores


def print_heatmap(sequence: str, saliency_scores: list, top_n: int = 10):
    """Print a text-based heatmap of saliency"""

    # Normalize scores
    scores = np.array([s[1] for s in saliency_scores])
    if scores.max() > 0:
        scores_norm = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores_norm = scores

    print("\n" + "=" * 60)
    print("SALIENCY HEATMAP")
    print("=" * 60)
    print("(Higher = more contribution to TOXIC verdict)\n")

    # Print sequence with intensity markers
    intensity_chars = " ░▒▓█"

    # Print in chunks of 50
    for start in range(0, len(sequence), 50):
        end = min(start + 50, len(sequence))
        chunk = sequence[start:end]
        intensities = scores_norm[start:end]

        # Position line
        print(f"{start+1:4d} ", end="")
        print(chunk)

        # Intensity line
        print("     ", end="")
        for intensity in intensities:
            idx = min(int(intensity * 4), 4)
            print(intensity_chars[idx], end="")
        print()
        print()

    # Top contributing residues
    print("=" * 60)
    print(f"TOP {top_n} CONTRIBUTING RESIDUES (to TOXIC score)")
    print("=" * 60)

    sorted_scores = sorted(saliency_scores, key=lambda x: x[1], reverse=True)
    for i, (residue, score, pos) in enumerate(sorted_scores[:top_n]):
        bar_len = int(score / scores.max() * 20) if scores.max() > 0 else 0
        bar = "█" * bar_len
        print(f"  {pos+1:3d} {residue}  {bar} ({score:.4f})")

    # Identify problematic regions (consecutive high-saliency)
    print("\n" + "=" * 60)
    print("PROBLEMATIC REGIONS (consecutive high saliency)")
    print("=" * 60)

    threshold = np.percentile(scores, 75)
    regions = []
    current_region = None

    for i, (residue, score, pos) in enumerate(saliency_scores):
        if score >= threshold:
            if current_region is None:
                current_region = {'start': pos, 'end': pos, 'residues': residue, 'scores': [score]}
            else:
                current_region['end'] = pos
                current_region['residues'] += residue
                current_region['scores'].append(score)
        else:
            if current_region and len(current_region['residues']) >= 3:
                regions.append(current_region)
            current_region = None

    if current_region and len(current_region['residues']) >= 3:
        regions.append(current_region)

    if regions:
        for i, region in enumerate(regions[:5]):
            avg_score = np.mean(region['scores'])
            print(f"\n  Region {i+1}: positions {region['start']+1}-{region['end']+1}")
            print(f"  Sequence: {region['residues']}")
            print(f"  Avg saliency: {avg_score:.4f}")
    else:
        print("  No concentrated problematic regions found.")


def main():
    parser = argparse.ArgumentParser(description="Attribution mapping for Assassin predictions")
    parser.add_argument("sequence", nargs="?", help="Amino acid sequence to analyze")
    parser.add_argument("--name", "-n", default="query", help="Name for the sequence")
    parser.add_argument("--top", "-t", type=int, default=10, help="Show top N residues")
    parser.add_argument("--pdb", help="Fetch sequence from PDB ID")

    args = parser.parse_args()

    # Get sequence
    if args.pdb:
        import requests
        url = f"https://www.rcsb.org/fasta/entry/{args.pdb}"
        response = requests.get(url)
        lines = response.text.strip().split('\n')
        sequence = ''.join(line for line in lines if not line.startswith('>'))
        args.name = f"PDB_{args.pdb}"
        print(f"Fetched {args.pdb}: {len(sequence)} AA")
    elif args.sequence:
        sequence = ''.join(c for c in args.sequence.upper() if c.isalpha())
    else:
        # Demo with 8FYV
        print("No sequence provided. Using PDB 8FYV as demo...")
        import requests
        url = "https://www.rcsb.org/fasta/entry/8FYV"
        response = requests.get(url)
        lines = response.text.strip().split('\n')
        sequence = ''.join(line for line in lines if not line.startswith('>'))[:200]
        args.name = "PDB_8FYV"

    print(f"\nAnalyzing: {args.name}")
    print(f"Length: {len(sequence)} AA")
    print(f"Sequence: {sequence[:50]}...")

    # Load model
    tokenizer, model, device = load_model()

    # First get the prediction
    model.eval()
    saprot_seq = to_saprot(sequence)
    inputs = tokenizer(saprot_seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    toxic_prob = probs[0].item()
    safe_prob = probs[1].item()

    print(f"\n{'='*60}")
    print("PREDICTION")
    print("="*60)
    print(f"Toxic: {toxic_prob*100:.1f}%")
    print(f"Safe:  {safe_prob*100:.1f}%")
    print(f"Verdict: {'TOXIC' if toxic_prob > 0.5 else 'SAFE'}")

    # Compute saliency (for toxic class)
    model.train()  # Enable gradients
    print("\nComputing saliency map...")
    saliency = compute_saliency(sequence, tokenizer, model, device, target_class=0)

    # Print heatmap
    print_heatmap(sequence, saliency, top_n=args.top)


if __name__ == "__main__":
    main()
