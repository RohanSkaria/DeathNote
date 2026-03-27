import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from peft import PeftModel
import torch.nn.functional as F

# 1. Define Paths (Update this if running locally)
BASE_MODEL_ID = "westlake-repl/SaProt_650M_AF2"
ADAPTER_PATH = "/content/drive/MyDrive/DeathNote/saprot_assassin_final" 

def load_assassin():
    print("🧬 Loading Base SaProt Architecture...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    base_model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL_ID, 
        num_labels=2, 
        torch_dtype=torch.bfloat16
    )
    
    print("🗡️ Equipping Assassin LoRA Weights...")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval() 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return tokenizer, model, device

def analyze_lineup(tokenizer, model, device):
    # The Test Lineup
    test_cases = {
        "1. THE CONTROL (Human Thioredoxin - Highly Soluble)": 
            "MVKQIESKTAFQEALDAAGDKLVVVDFSATWCGPCKMIKPFFHSLSEKYSNVIFLEVDVDDCQDVASECEVKCMPTFQFFKKGQKVGEFSGANKEKLEATINELV",
            
        "2. THE BAD (Amyloid-Beta 42 - Severe Aggregator)": 
            "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA",
            
        "3. THE DECEPTIVE (AlphaFold Hallucination - Stealth Failure)": 
            "MDELYKVGALSKGQLKEFLDANLAGSGSGMDELYKMSDKIIHLTDDSFDTDVLKADGAILVDFWAEWCGPCKMIAPILDEIADEYQGKLTVAKLNIDQNPGTAPKYGIRGIPTLLLFKNGEVAAT"
    }

    print("\n" + "="*50)
    print("🎯 INITIATING ASSASSIN INFERENCE PROTOCOL")
    print("="*50)

    for name, sequence in test_cases.items():
        # Apply the SaProt '#' mask trick
        saprot_seq = "".join([aa + "#" for aa in sequence])
        
        inputs = tokenizer(saprot_seq, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            
        # Class 0 is Toxic/Failure, Class 1 is Safe/Functional
        prob_toxic = probs[0][0].item() * 100
        prob_safe = probs[0][1].item() * 100
        
        print(f"\n{name}")
        print(f"Sequence Length: {len(sequence)} AAs")
        print(f"Toxic Probability: {prob_toxic:>6.2f}%")
        print(f"Safe Probability:  {prob_safe:>6.2f}%")
        
        if prob_toxic > 50:
            print("🚨 VERDICT: REJECTED (Predicted Misfold/Aggregation)")
        else:
            print("✅ VERDICT: CLEARED (Predicted Stable)")

# --- EXECUTION ---
if __name__ == "__main__":
    tokenizer, model, device = load_assassin()
    analyze_lineup(tokenizer, model, device)