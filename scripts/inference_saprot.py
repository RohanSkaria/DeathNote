#!/usr/bin/env python3
"""
inference_saprot.py - Run inference with trained SaProt Assassin model

Usage:
    python inference_saprot.py "MKTAYIAKQRQISFVKSHFSRQLE..."
    python inference_saprot.py --file sequences.txt
"""

import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

MODEL_PATH = "/content/drive/MyDrive/DeathNote/saprot_assassin"
BASE_MODEL = "westlake-repl/SaProt_650M_AF2"


def to_saprot(sequence: str) -> str:
    """Convert amino acid sequence to SaProt format with # masks"""
    s = str(sequence).upper().strip()
    return "#".join(list(s)) + "#"


def load_model(model_path: str = MODEL_PATH):
    """Load the trained SaProt + LoRA model"""
    print("Loading SaProt Assassin model...")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    # Load base model + LoRA adapters
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, tokenizer, device


def predict(sequence: str, model, tokenizer, device) -> dict:
    """
    Predict safety score for a protein sequence.

    Returns:
        dict with 'safe_prob', 'toxic_prob', 'prediction', 'confidence'
    """
    # Convert to SaProt format
    saprot_seq = to_saprot(sequence)

    # Tokenize
    inputs = tokenizer(
        saprot_seq,
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)

    # Inference
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    toxic_prob = probs[0].item()
    safe_prob = probs[1].item()

    prediction = "SAFE" if safe_prob > toxic_prob else "TOXIC"
    confidence = max(safe_prob, toxic_prob)

    return {
        "prediction": prediction,
        "confidence": confidence,
        "safe_prob": safe_prob,
        "toxic_prob": toxic_prob,
        "sequence_length": len(sequence)
    }


def main():
    parser = argparse.ArgumentParser(description="SaProt Assassin Inference")
    parser.add_argument("sequence", nargs="?", help="Amino acid sequence to classify")
    parser.add_argument("--file", "-f", help="File with sequences (one per line)")
    parser.add_argument("--model", "-m", default=MODEL_PATH, help="Model path")
    args = parser.parse_args()

    model, tokenizer, device = load_model(args.model)

    sequences = []
    if args.file:
        with open(args.file) as f:
            sequences = [line.strip() for line in f if line.strip()]
    elif args.sequence:
        sequences = [args.sequence]
    else:
        # Demo sequence
        sequences = ["MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAAALEHHHHHH"]

    print("\n" + "=" * 60)
    print("SAPROT ASSASSIN - PROTEIN SAFETY PREDICTION")
    print("=" * 60)

    for i, seq in enumerate(sequences):
        result = predict(seq, model, tokenizer, device)

        print(f"\nSequence {i+1} ({result['sequence_length']} AA):")
        print(f"  {seq[:50]}{'...' if len(seq) > 50 else ''}")
        print(f"  Prediction: {result['prediction']}")
        print(f"  Confidence: {result['confidence']:.1%}")
        print(f"  Safe:  {result['safe_prob']:.1%}")
        print(f"  Toxic: {result['toxic_prob']:.1%}")

        # Flag potential hallucinations
        if result['prediction'] == "TOXIC" and result['confidence'] > 0.8:
            print("  ⚠️  HIGH-CONFIDENCE TOXICITY FLAG")


if __name__ == "__main__":
    main()
