"""
Death Note - SaProt Assassin
Protein Hallucination Detector

A structure-aware classifier that detects potentially unstable or
"hallucinated" protein designs that look good on paper but may fail in the lab.
"""

import streamlit as st
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import pandas as pd
import os
import io

# === CONFIG ===
BASE_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
ADAPTER_PATH = os.path.join(os.path.dirname(__file__), "saprot_assassin")

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Death Note - Protein Assassin",
    page_icon="🗡️",
    layout="wide"
)

# === CACHED MODEL LOADING ===
@st.cache_resource
def load_model():
    """Load SaProt + LoRA model (cached)"""
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID,
        num_labels=2,
        torch_dtype=torch.float32  # Use float32 for CPU compatibility
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return tokenizer, model, device


def to_saprot(sequence: str) -> str:
    """Convert AA sequence to SaProt format with # structural masks"""
    return "#".join(list(str(sequence).upper().strip())) + "#"


def predict(sequence: str, tokenizer, model, device) -> dict:
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
        "toxic_prob": toxic_prob,
        "safe_prob": safe_prob,
        "verdict": "TOXIC" if toxic_prob > 0.5 else "SAFE",
        "confidence": max(toxic_prob, safe_prob),
        "length": len(sequence)
    }


# === UI ===
st.title("🗡️ Death Note - Protein Assassin")
st.markdown("""
**Structure-aware hallucination detector for protein designs.**

This model identifies protein sequences that may look valid but could fail during
expression, folding, or manufacturing. Trained on ProteinGym stability data with
SaProt (structure-aware) embeddings.
""")

# Load model
with st.spinner("Loading SaProt Assassin model..."):
    try:
        tokenizer, model, device = load_model()
        st.success(f"Model loaded on {device}")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

# === TABS ===
tab1, tab2 = st.tabs(["🔬 Single Sequence", "📊 Batch Analysis"])

# === SINGLE SEQUENCE TAB ===
with tab1:
    st.subheader("Analyze a Single Protein Sequence")

    sequence_input = st.text_area(
        "Paste amino acid sequence:",
        height=150,
        placeholder="MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGD..."
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        analyze_btn = st.button("🎯 Analyze", type="primary")

    if analyze_btn and sequence_input:
        # Clean sequence
        sequence = ''.join(c for c in sequence_input.upper() if c.isalpha())

        if len(sequence) < 10:
            st.error("Sequence too short (minimum 10 amino acids)")
        else:
            with st.spinner("Running inference..."):
                result = predict(sequence, tokenizer, model, device)

            # Display results
            st.markdown("---")

            col1, col2, col3 = st.columns(3)

            with col1:
                if result["verdict"] == "TOXIC":
                    st.error(f"## 🚨 {result['verdict']}")
                else:
                    st.success(f"## ✅ {result['verdict']}")

            with col2:
                st.metric("Confidence", f"{result['confidence']*100:.1f}%")

            with col3:
                st.metric("Length", f"{result['length']} AA")

            # Probability bars
            st.markdown("### Probability Breakdown")
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Safe Probability**")
                st.progress(result["safe_prob"])
                st.write(f"{result['safe_prob']*100:.1f}%")

            with col2:
                st.markdown("**Toxic Probability**")
                st.progress(result["toxic_prob"])
                st.write(f"{result['toxic_prob']*100:.1f}%")

            # Interpretation
            st.markdown("### Interpretation")
            if result["verdict"] == "TOXIC":
                if result["confidence"] > 0.9:
                    st.warning("""
                    **High-confidence toxicity flag.** This sequence has structural features
                    associated with expression failure, aggregation, or misfolding.
                    Consider redesigning problematic regions.
                    """)
                else:
                    st.info("""
                    **Moderate toxicity flag.** Some concerning features detected.
                    May still express but warrants careful validation.
                    """)
            else:
                if result["confidence"] > 0.9:
                    st.success("""
                    **High-confidence safe prediction.** Sequence has structural features
                    consistent with stable, expressible proteins.
                    """)
                else:
                    st.info("""
                    **Moderate confidence.** Sequence appears okay but consider
                    experimental validation.
                    """)

# === BATCH ANALYSIS TAB ===
with tab2:
    st.subheader("Batch Analysis")
    st.markdown("Upload a CSV or FASTA file with multiple sequences.")

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "fasta", "fa", "txt"],
        help="CSV should have 'sequence' or 'aa_sequence' column. FASTA format also supported."
    )

    if uploaded_file:
        # Parse file
        sequences = []
        names = []

        content = uploaded_file.read().decode('utf-8')

        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(io.StringIO(content))
            seq_col = None
            for col in ['aa_sequence', 'sequence', 'full_sequence', 'seq']:
                if col in df.columns:
                    seq_col = col
                    break
            if seq_col:
                sequences = df[seq_col].tolist()
                if 'name' in df.columns:
                    names = df['name'].tolist()
                elif 'sequence_id' in df.columns:
                    names = df['sequence_id'].tolist()
                else:
                    names = [f"seq_{i}" for i in range(len(sequences))]
        else:
            # FASTA format
            current_name = None
            current_seq = []
            for line in content.split('\n'):
                line = line.strip()
                if line.startswith('>'):
                    if current_name and current_seq:
                        sequences.append(''.join(current_seq))
                        names.append(current_name)
                    current_name = line[1:].split()[0]
                    current_seq = []
                elif line:
                    current_seq.append(line)
            if current_name and current_seq:
                sequences.append(''.join(current_seq))
                names.append(current_name)

        st.write(f"Found **{len(sequences)}** sequences")

        if st.button("🚀 Run Batch Analysis", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i, (name, seq) in enumerate(zip(names, sequences)):
                seq_clean = ''.join(c for c in str(seq).upper() if c.isalpha())
                if len(seq_clean) >= 10:
                    result = predict(seq_clean, tokenizer, model, device)
                    result['name'] = name
                    result['sequence'] = seq_clean[:50] + "..." if len(seq_clean) > 50 else seq_clean
                    results.append(result)

                progress_bar.progress((i + 1) / len(sequences))
                status_text.text(f"Processing {i+1}/{len(sequences)}...")

            status_text.text("Done!")

            # Create results dataframe
            results_df = pd.DataFrame(results)
            results_df = results_df.sort_values('toxic_prob', ascending=False)
            results_df['rank'] = range(1, len(results_df) + 1)

            # Summary stats
            st.markdown("---")
            st.markdown("### Summary")

            col1, col2, col3 = st.columns(3)
            toxic_count = (results_df['verdict'] == 'TOXIC').sum()
            safe_count = (results_df['verdict'] == 'SAFE').sum()

            with col1:
                st.metric("Total Sequences", len(results_df))
            with col2:
                st.metric("🚨 Toxic", toxic_count)
            with col3:
                st.metric("✅ Safe", safe_count)

            # Results table
            st.markdown("### Results (Ranked by Toxicity)")
            display_df = results_df[['rank', 'name', 'length', 'toxic_prob', 'safe_prob', 'verdict', 'confidence']]
            display_df.columns = ['Rank', 'Name', 'Length', 'Toxic %', 'Safe %', 'Verdict', 'Confidence']
            display_df['Toxic %'] = (display_df['Toxic %'] * 100).round(1)
            display_df['Safe %'] = (display_df['Safe %'] * 100).round(1)
            display_df['Confidence'] = (display_df['Confidence'] * 100).round(1)

            st.dataframe(display_df, use_container_width=True)

            # Download button
            csv = results_df.to_csv(index=False)
            st.download_button(
                "📥 Download Results CSV",
                csv,
                "assassin_results.csv",
                "text/csv"
            )

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
    **Death Note** is a protein safety classifier that detects
    potentially problematic protein designs before they fail in the lab.

    ### How it works
    1. Converts sequence to SaProt format (structure-aware tokens)
    2. Runs through fine-tuned SaProt-650M model
    3. Predicts probability of structural failure

    ### Training Data
    - ProteinGym stability assays
    - ESOL expression data
    - Aggregation-prone sequences
    - UniProt wild-type controls

    ### Limitations
    - Optimized for 200-600 AA proteins
    - Human-centric bias (may over-flag bacterial/viral proteins)
    - Cannot detect all failure modes

    ### Links
    - [GitHub](https://github.com/RohanSkaria/DeathNote)
    - [ProteinGym](https://proteingym.org)
    """)

    st.markdown("---")
    st.markdown("Made with SaProt + LoRA")
