"""
Protein Safety Classifier - Streamlit Web App
Clean web interface for protein sequence safety prediction
"""

import streamlit as st
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os
import streamlit.components.v1 as components

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Protein Sequence Safety Filter",
    page_icon="🧬",
    layout="wide"
)

# --- MODEL DEFINITION (Must match training script exactly) ---
class SafetyClassifier(nn.Module):
    def __init__(self, model_name, fine_tune_last_layer=True):
        super(SafetyClassifier, self).__init__()
        self.esm = AutoModel.from_pretrained(model_name)
        
        # FREEZE most of the model
        for param in self.esm.parameters():
            param.requires_grad = False
        
        # UNFREEZE the last Transformer Layer (Fine-Tuning Strategy)
        if fine_tune_last_layer and hasattr(self.esm, 'encoder') and hasattr(self.esm.encoder, 'layer'):
            for param in self.esm.encoder.layer[-1].parameters():
                param.requires_grad = True
        
        # Also unfreeze the pooler if it exists
        if hasattr(self.esm, 'pooler') and self.esm.pooler is not None:
            for param in self.esm.pooler.parameters():
                param.requires_grad = True
        
        # UPGRADED Classifier Head (matches training script)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.esm.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1)
        )

    def forward(self, input_ids, attention_mask):
        output = self.esm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = output.last_hidden_state[:, 0, :]
        return self.classifier(pooled_output)

# --- LOAD MODEL (Cached for speed) ---
@st.cache_resource
def load_pipeline():
    model_name = "facebook/esm2_t12_35M_UR50D"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize model
    model = SafetyClassifier(model_name, fine_tune_last_layer=True)
    
    # Load weights
    model_path = "best_model_state.bin"
    if not os.path.exists(model_path):
        st.error(f"❌ Model file '{model_path}' not found. Please place it in the same folder.")
        st.info(f"Current directory: {os.getcwd()}")
        st.info(f"Files in directory: {', '.join(os.listdir('.'))}")
        return None, None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            best_mcc = checkpoint.get('best_mcc', 'N/A')
        else:
            model.load_state_dict(checkpoint)
            best_mcc = 'N/A'
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, None, None
        
    model.eval()
    return tokenizer, model, best_mcc

# --- VALIDATION ---
def validate_sequence(sequence):
    """Validate protein sequence contains only valid amino acids"""
    valid_aas = set('ACDEFGHIKLMNPQRSTVWY')
    sequence_clean = ''.join(sequence.split()).upper()
    
    invalid_chars = set(sequence_clean) - valid_aas
    if invalid_chars:
        return False, f"Invalid amino acid characters found: {', '.join(sorted(invalid_chars))}"
    
    if len(sequence_clean) < 10:
        return False, "Sequence too short (minimum 10 amino acids)"
    
    if len(sequence_clean) > 512:
        return False, f"Sequence too long ({len(sequence_clean)} > 512 amino acids)"
    
    return True, sequence_clean

# --- UI LAYOUT ---
st.title("🧬 Protein Safety Classifier")
st.markdown("""
**Prototype v0.1** | Fine-Tuned ESM-2 (35M)  
This tool predicts the **functional viability** of protein sequences based on Deep Mutational Scanning data.
""")

# Load model
tokenizer, model, best_mcc = load_pipeline()

if model is None or tokenizer is None:
    st.stop()

# Display model info
with st.sidebar:
    st.header("Model Information")
    st.info(f"**Best MCC:** {best_mcc:.4f}" if isinstance(best_mcc, float) else f"**Best MCC:** {best_mcc}")
    st.caption("Trained on ProteinGym DMS data")
    st.caption("2.47M sequences | 217 assays")

# Input section
st.header("Input Sequence")

# Initialize session state for sequence
if 'input_sequence' not in st.session_state:
    st.session_state.input_sequence = ""

# Example sequences
with st.expander("Example Sequences"):
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Safe (Cytochrome C)")
        example_safe = "MGDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFVGIKKKEERADLIAYLKKATNE"
        # Display sequence in a copyable code block
        st.code(example_safe, language=None)
        
        # Custom HTML button that copies on click (using template literals)
        copy_button_html = f"""
        <button onclick="navigator.clipboard.writeText(`{example_safe}`).then(() => {{
            const btn = event.target;
            const originalText = btn.textContent;
            const originalBg = btn.style.backgroundColor;
            btn.textContent = 'Copied!';
            btn.style.backgroundColor = '#28a745';
            btn.style.color = 'white';
            setTimeout(() => {{
                btn.textContent = originalText;
                btn.style.backgroundColor = '#0d6efd';
                btn.style.color = 'white';
            }}, 2000);
        }}).catch(err => alert('Copy failed. Please select and copy manually.'));" 
        style="padding: 0.5rem 1rem; background-color: #0d6efd; color: white; border: none; border-radius: 0.25rem; cursor: pointer; font-size: 0.875rem; font-weight: 500;">
        Copy Safe Example
        </button>
        """
        components.html(copy_button_html, height=40)
    
    with col2:
        st.subheader("Toxic (HIV)")
        example_toxic = "MGGKWSKSSVIGWPTVRERMRRAEPAADGVGAASRDLEKHGAITSSNTAATNAACAWLEAQEEEEVGFPVTPQVPLRPMTYKAAVDLSHFLKEKGGLEGLIHSQRRQDILDLWIYHTQGYFPDWQNYTPGPGVRYPLTFGWLYKLVPVEPEKVEEANKGENTSLLHPVSLHGMDDPEREVL"
        # Display sequence in a copyable code block
        st.code(example_toxic, language=None)
        
        # Custom HTML button that copies on click (using template literals)
        copy_button_html_toxic = f"""
        <button onclick="navigator.clipboard.writeText(`{example_toxic}`).then(() => {{
            const btn = event.target;
            const originalText = btn.textContent;
            const originalBg = btn.style.backgroundColor;
            btn.textContent = 'Copied!';
            btn.style.backgroundColor = '#28a745';
            btn.style.color = 'white';
            setTimeout(() => {{
                btn.textContent = originalText;
                btn.style.backgroundColor = '#0d6efd';
                btn.style.color = 'white';
            }}, 2000);
        }}).catch(err => alert('Copy failed. Please select and copy manually.'));" 
        style="padding: 0.5rem 1rem; background-color: #0d6efd; color: white; border: none; border-radius: 0.25rem; cursor: pointer; font-size: 0.875rem; font-weight: 500;">
        Copy Toxic Example
        </button>
        """
        components.html(copy_button_html_toxic, height=40)

sequence = st.text_area(
    "Enter Protein Sequence:",
    value=st.session_state.input_sequence,
    height=150,
    placeholder="MEEPQSDPSVEPPLSQETFSDLWKLLPENNVLSPLPSQAMDDLMLSPDDIEQWFTEDPG...",
    help="Paste your protein sequence (amino acids: A, C, D, E, F, G, H, I, K, L, M, N, P, Q, R, S, T, V, W, Y)",
    key="sequence_input"
)

# Sync session state with text_area value
st.session_state.input_sequence = sequence

# Analyze button
if st.button("Analyze Sequence", type="primary", use_container_width=True):
    if not sequence or len(sequence.strip()) == 0:
        st.warning("⚠️ Please enter a protein sequence.")
    else:
        # Validate sequence
        is_valid, result = validate_sequence(sequence)
        
        if not is_valid:
            st.error(f"❌ {result}")
        else:
            clean_sequence = result
            
            # Inference
            with st.spinner("Analyzing protein structure..."):
                inputs = tokenizer(
                    clean_sequence,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512,
                    padding='max_length'
                )
                
                with torch.no_grad():
                    logits = model(inputs['input_ids'], inputs['attention_mask'])
                    prob = torch.sigmoid(logits).item()
                    prediction = 1 if prob > 0.5 else 0
            
            # Display Results
            st.header("📊 Results")
            
            # --- OPTIMAL THRESHOLDS BASED ON YOUR DATA ---
            FAIL_THRESHOLD = 0.50  # Below this = TOXIC
            PASS_THRESHOLD = 0.75  # Above this = SAFE
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Safety Probability", f"{prob:.4f}")
                st.caption("Higher = Safer")
            
            with col2:
                if prob >= PASS_THRESHOLD:
                    st.success("✅ **Verdict: SAFE (High Confidence)**")
                    st.caption(f"Score {prob:.4f} is excellent. Proceed.")
                elif prob > FAIL_THRESHOLD:
                    st.warning("🟡 **Verdict: UNCERTAIN (Review Required)**")
                    st.caption(f"Score {prob:.4f} passes, but is low-confidence. Check structure manually.")
                else:
                    st.error("❌ **Verdict: TOXIC (Rejected)**")
                    st.caption(f"Score {prob:.4f} is below the viability threshold.")
            
            with col3:
                if prob >= PASS_THRESHOLD:
                    confidence = "High"
                elif prob > FAIL_THRESHOLD:
                    confidence = "Medium"
                else:
                    confidence = "High"
                
                st.metric("Confidence", confidence)
                st.caption("Prediction reliability")
            
            # Visual progress bar with thresholds
            st.markdown("### Safety Score")
            if prob >= PASS_THRESHOLD:
                st.progress(prob)
                st.caption(f"{prob*100:.1f}% - SAFE (High Confidence)")
            elif prob > FAIL_THRESHOLD:
                st.progress(prob)
                st.caption(f"{prob*100:.1f}% - UNCERTAIN (Review Required)")
            else:
                st.progress(prob)
                st.caption(f"{prob*100:.1f}% - TOXIC (Rejected)")
            
            # Threshold indicators
            st.markdown("**Thresholds:**")
            col_thresh1, col_thresh2, col_thresh3 = st.columns(3)
            with col_thresh1:
                st.caption(f"🔴 TOXIC: < {FAIL_THRESHOLD}")
            with col_thresh2:
                st.caption(f"🟡 UNCERTAIN: {FAIL_THRESHOLD} - {PASS_THRESHOLD}")
            with col_thresh3:
                st.caption(f"🟢 SAFE: > {PASS_THRESHOLD}")
            
            # Additional info
            with st.expander("🔬 Technical Details"):
                st.json({
                    "Model Architecture": "ESM-2 (35M) + Fine-tuned last layer + MLP Head",
                    "Sequence Length": len(clean_sequence),
                    "Raw Logits": f"{logits.item():.4f}",
                    "Probability": f"{prob:.4f}",
                    "Prediction": "Safe" if prediction == 1 else "Toxic",
                    "Valid Amino Acids": True
                })
            
            # Interpretation guide
            st.markdown("---")
            st.markdown("### 📖 Interpretation Guide")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.success("""
                **Score ≥ 0.75 (SAFE)**
                - High confidence prediction
                - Protein is likely functional
                - Proceed with experimental validation
                """)
            
            with col2:
                st.warning("""
                **Score 0.50 - 0.75 (UNCERTAIN)**
                - Low-confidence prediction
                - Requires manual review
                - Check structure and properties
                """)
            
            with col3:
                st.error("""
                **Score < 0.50 (TOXIC)**
                - High confidence rejection
                - Protein likely non-functional
                - Consider redesign or further analysis
                """)

# Footer
st.markdown("---")
st.caption("Built with Streamlit | Model trained on ProteinGym DMS benchmark data")

