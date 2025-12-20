"""
Protein Safety Classifier - Inference Script
Loads trained model and makes predictions on protein sequences
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import os

# Configuration
MODEL_NAME = "facebook/esm2_t12_35M_UR50D"
MODEL_PATH = "best_model_state.bin"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🔧 Device: {DEVICE}")
print(f"📦 Model: {MODEL_NAME}")

# --- MODEL DEFINITION (Must match training script) ---
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
        
        # UPGRADED Classifier Head
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

# --- LOAD MODEL ---
print(f"\n📂 Loading model from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"❌ Error: Model file not found at {MODEL_PATH}")
    print(f"   Current directory: {os.getcwd()}")
    print(f"   Files in directory: {os.listdir('.')}")
    exit(1)

# Initialize model
model = SafetyClassifier(MODEL_NAME, fine_tune_last_layer=True)
model = model.to(DEVICE)

# Load checkpoint (weights_only=False for PyTorch 2.6+ compatibility)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Model loaded (Best MCC: {checkpoint.get('best_mcc', 'N/A')})")
else:
    # Fallback: assume it's just the state dict
    model.load_state_dict(checkpoint)
    print("✅ Model loaded")

model.eval()

# Load tokenizer
print(f"🔤 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
print("✅ Ready for inference!\n")

# --- INFERENCE FUNCTION ---
def predict_safety(sequence):
    """
    Predict if a protein sequence is safe (1) or toxic (0)
    
    Args:
        sequence: Protein sequence string (amino acids)
    
    Returns:
        dict with prediction, probability, and safety status
    """
    # Handle empty sequences
    if not sequence or len(sequence.strip()) == 0:
        return {
            'prediction': 0,
            'probability': 0.0,
            'safety_status': 'INVALID',
            'confidence': 'N/A'
        }
    
    # Clean sequence (remove whitespace, newlines)
    sequence = ''.join(sequence.split()).upper()
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        sequence,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(DEVICE)
    attention_mask = encoding['attention_mask'].to(DEVICE)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probability = torch.sigmoid(outputs).item()
        prediction = 1 if probability > 0.5 else 0
    
    # Interpret results
    if prediction == 1:
        safety_status = "SAFE"
        confidence = "High" if probability > 0.8 else "Medium" if probability > 0.6 else "Low"
    else:
        safety_status = "TOXIC"
        confidence = "High" if probability < 0.2 else "Medium" if probability < 0.4 else "Low"
    
    return {
        'prediction': prediction,
        'probability': probability,
        'safety_status': safety_status,
        'confidence': confidence
    }

# --- INTERACTIVE MODE ---
if __name__ == '__main__':
    print("="*80)
    print("🧬 Protein Safety Classifier - Inference Mode")
    print("="*80)
    print("\nEnter protein sequences to classify.")
    print("Type 'quit' or 'exit' to stop.\n")
    
    # Test with known toxic sequence
    test_sequence = "MGGKWSKSSVIGWPTVRERMRRAEPAADGVGAASRDLEKHGAITSSNTAATNAACAWLEAQEEEEVGFPVTPQVPLRPMTYKAAVDLSHFLKEKGGLEGLIHSQRRQDILDLWIYHTQGYFPDWQNYTPGPGVRYPLTFGWLYKLVPVEPEKVEEANKGENTSLLHPVSLHGMDDPEREVL"
    print("💡 Example: Testing with known toxic HIV sequence...")
    result = predict_safety(test_sequence)
    print(f"\n   Sequence: {test_sequence[:50]}...")
    print(f"   Prediction: {result['safety_status']} (Label: {result['prediction']})")
    print(f"   Probability: {result['probability']:.4f}")
    print(f"   Confidence: {result['confidence']}\n")
    print("-"*80)
    
    # Interactive loop
    while True:
        try:
            sequence = input("\n🧬 Enter protein sequence (or 'quit'): ").strip()
            
            if sequence.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            if not sequence:
                print("⚠️  Please enter a sequence")
                continue
            
            # Predict
            result = predict_safety(sequence)
            
            # Display results
            print(f"\n📊 Results:")
            print(f"   Safety Status: {result['safety_status']}")
            print(f"   Label: {result['prediction']} ({'Safe' if result['prediction'] == 1 else 'Toxic'})")
            print(f"   Probability: {result['probability']:.4f}")
            print(f"   Confidence: {result['confidence']}")
            
            # Visual indicator
            if result['safety_status'] == 'TOXIC':
                print(f"   ⚠️  WARNING: This protein may be toxic!")
            else:
                print(f"   ✅ This protein appears safe")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("   Please try again with a valid protein sequence")

