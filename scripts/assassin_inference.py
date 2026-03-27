#!/usr/bin/env python3
"""
assassin_inference.py - Production inference for SaProt Assassin

Features:
- Single sequence or batch inference
- CSV/FASTA file input
- Ranked output by toxicity
- Works locally or in Colab
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import pandas as pd
import argparse
import os
import sys

# === CONFIG ===
BASE_MODEL_ID = "westlake-repl/SaProt_650M_AF2"

# Auto-detect environment
if os.path.exists("/content/drive/MyDrive"):
    # Colab
    DEFAULT_ADAPTER_PATH = "/content/drive/MyDrive/DeathNote/saprot_assassin"
else:
    # Local - check common locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    DEFAULT_ADAPTER_PATH = os.path.join(project_dir, "saprot_assassin")


def to_saprot(sequence: str) -> str:
    """Convert AA sequence to SaProt format with # structural masks"""
    return "#".join(list(str(sequence).upper().strip())) + "#"


def load_model(adapter_path: str = None):
    """Load SaProt + LoRA Assassin model"""
    adapter_path = adapter_path or DEFAULT_ADAPTER_PATH

    print("Loading SaProt Assassin...")
    print(f"  Base: {BASE_MODEL_ID}")
    print(f"  Adapter: {adapter_path}")

    if not os.path.exists(adapter_path):
        raise FileNotFoundError(
            f"Adapter not found: {adapter_path}\n"
            "Download from Google Drive or update DEFAULT_ADAPTER_PATH"
        )

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
        torch_dtype=torch.bfloat16
    )

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    print(f"  Device: {device}")
    return tokenizer, model, device


def predict_single(sequence: str, tokenizer, model, device) -> dict:
    """Predict toxicity for a single sequence"""
    saprot_seq = to_saprot(sequence)

    inputs = tokenizer(
        saprot_seq,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)[0]

    toxic_prob = probs[0].item()
    safe_prob = probs[1].item()

    return {
        "sequence": sequence[:50] + "..." if len(sequence) > 50 else sequence,
        "length": len(sequence),
        "toxic_prob": toxic_prob,
        "safe_prob": safe_prob,
        "verdict": "TOXIC" if toxic_prob > 0.5 else "SAFE",
        "confidence": max(toxic_prob, safe_prob)
    }


def predict_batch(sequences: list, tokenizer, model, device, names: list = None) -> pd.DataFrame:
    """Predict toxicity for multiple sequences, return ranked DataFrame"""
    results = []

    for i, seq in enumerate(sequences):
        name = names[i] if names else f"seq_{i+1}"
        result = predict_single(seq, tokenizer, model, device)
        result["name"] = name
        results.append(result)

        if (i + 1) % 10 == 0:
            print(f"  Processed {i+1}/{len(sequences)}...")

    df = pd.DataFrame(results)
    # Rank by toxicity (most toxic first)
    df = df.sort_values("toxic_prob", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)

    return df[["rank", "name", "length", "toxic_prob", "safe_prob", "verdict", "confidence", "sequence"]]


def load_sequences_from_file(filepath: str) -> tuple:
    """Load sequences from CSV or FASTA file"""
    sequences = []
    names = []

    if filepath.endswith(".csv"):
        df = pd.read_csv(filepath)
        # Try common column names
        seq_col = None
        for col in ["aa_sequence", "sequence", "full_sequence", "seq"]:
            if col in df.columns:
                seq_col = col
                break
        if not seq_col:
            raise ValueError(f"No sequence column found. Columns: {df.columns.tolist()}")

        sequences = df[seq_col].tolist()
        if "name" in df.columns:
            names = df["name"].tolist()
        elif "sequence_id" in df.columns:
            names = df["sequence_id"].tolist()
        else:
            names = [f"seq_{i}" for i in range(len(sequences))]

    elif filepath.endswith(".fasta") or filepath.endswith(".fa"):
        current_name = None
        current_seq = []

        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if current_name:
                        sequences.append("".join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].split()[0]
                    current_seq = []
                else:
                    current_seq.append(line)

            if current_name:
                sequences.append("".join(current_seq))
                names.append(current_name)

    else:
        # Plain text, one sequence per line
        with open(filepath) as f:
            sequences = [line.strip() for line in f if line.strip()]
        names = [f"seq_{i}" for i in range(len(sequences))]

    return sequences, names


def main():
    parser = argparse.ArgumentParser(
        description="SaProt Assassin - Protein Toxicity Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single sequence
  python assassin_inference.py "MKTAYIAKQRQ..."

  # Batch from file
  python assassin_inference.py --file designs.csv --output ranked.csv

  # Run demo
  python assassin_inference.py --demo
        """
    )
    parser.add_argument("sequence", nargs="?", help="Single AA sequence to analyze")
    parser.add_argument("--file", "-f", help="CSV/FASTA file with sequences")
    parser.add_argument("--output", "-o", help="Output CSV for batch results")
    parser.add_argument("--adapter", "-a", help="Path to LoRA adapter")
    parser.add_argument("--demo", action="store_true", help="Run demo with test sequences")
    parser.add_argument("--top", "-n", type=int, default=10, help="Show top N results (default: 10)")

    args = parser.parse_args()

    # Load model
    tokenizer, model, device = load_model(args.adapter)

    print("\n" + "=" * 60)
    print("SAPROT ASSASSIN - STRUCTURAL TOXICITY DETECTOR")
    print("=" * 60)

    if args.demo:
        # Demo mode with test cases
        test_cases = {
            "Control_Thioredoxin": "MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDDCQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV",
            "Bad_AmyloidBeta42": "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
            "Deceptive_AFHallucination": "MDELYKVGALSKGQLKEFLDANLAGSGSGMDELYKMSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAAT"
        }
        sequences = list(test_cases.values())
        names = list(test_cases.keys())

        df = predict_batch(sequences, tokenizer, model, device, names)

        print("\n=== DEMO RESULTS ===\n")
        for _, row in df.iterrows():
            print(f"{row['name']}")
            print(f"  Length: {row['length']} AA")
            print(f"  Toxic:  {row['toxic_prob']*100:.1f}%")
            print(f"  Safe:   {row['safe_prob']*100:.1f}%")
            print(f"  Verdict: {row['verdict']}")
            print()

    elif args.file:
        # Batch mode
        print(f"\nLoading sequences from: {args.file}")
        sequences, names = load_sequences_from_file(args.file)
        print(f"Found {len(sequences)} sequences")

        df = predict_batch(sequences, tokenizer, model, device, names)

        # Show top N
        print(f"\n=== TOP {min(args.top, len(df))} MOST TOXIC ===\n")
        for _, row in df.head(args.top).iterrows():
            print(f"#{row['rank']} {row['name']}: {row['toxic_prob']*100:.1f}% toxic ({row['length']} AA)")

        # Save if output specified
        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nFull results saved to: {args.output}")

        # Summary
        toxic_count = (df["verdict"] == "TOXIC").sum()
        print(f"\n=== SUMMARY ===")
        print(f"Total: {len(df)}")
        print(f"Toxic: {toxic_count} ({toxic_count/len(df)*100:.1f}%)")
        print(f"Safe:  {len(df)-toxic_count} ({(len(df)-toxic_count)/len(df)*100:.1f}%)")

    elif args.sequence:
        # Single sequence mode
        result = predict_single(args.sequence, tokenizer, model, device)

        print(f"\nSequence: {result['sequence']}")
        print(f"Length:   {result['length']} AA")
        print(f"Toxic:    {result['toxic_prob']*100:.1f}%")
        print(f"Safe:     {result['safe_prob']*100:.1f}%")
        print(f"Verdict:  {result['verdict']} ({result['confidence']*100:.1f}% confidence)")

        if result["verdict"] == "TOXIC" and result["confidence"] > 0.8:
            print("\n⚠️  HIGH-CONFIDENCE TOXICITY FLAG")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
