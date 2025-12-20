# Protein Safety Classifier - Visualization & Inference

This folder contains scripts for visualizing training results and running inference on protein sequences.

## Files

- `training_history.csv` - Training metrics from model training
- `best_model_state.bin` - Trained model weights
- `visualize_training.py` - Creates visualization plots
- `inference.py` - Interactive inference script for predicting protein safety
- `app.py` - Streamlit web application
- `run_app.sh` - Helper script to run the Streamlit app

## Setup

### For Visualization

Install required libraries:
```bash
pip install matplotlib seaborn pandas
```

### For Inference

Install required libraries:
```bash
pip install torch transformers pandas
```

### For Web App

Install Streamlit:
```bash
pip install streamlit
```

Or install everything:
```bash
pip install torch transformers matplotlib seaborn pandas streamlit
```

## Usage

### 1. Visualize Training Results

```bash
python visualize_training.py
```

This creates two files:
- `results_graph.png` - Comprehensive 6-panel dashboard
- `results_graph_simple.png` - Simple 2-panel version

### 2. Run Inference (Command Line)

```bash
python inference.py
```

Then enter protein sequences when prompted. The script will:
- Load the trained model
- Tokenize your sequence
- Predict safety (Safe/Toxic)
- Show probability and confidence

**Example test sequence** (known toxic HIV protein):
```
MGGKWSKSSVIGWPTVRERMRRAEPAADGVGAASRDLEKHGAITSSNTAATNAACAWLEAQEEEEVGFPVTPQVPLRPMTYKAAVDLSHFLKEKGGLEGLIHSQRRQDILDLWIYHTQGYFPDWQNYTPGPGVRYPLTFGWLYKLVPVEPEKVEEANKGENTSLLHPVSLHGMDDPEREVL
```

Expected output: **TOXIC** (probability close to 0.0)

### 3. Run Web App (Streamlit)

**Option 1: Using the helper script**
```bash
./run_app.sh
```

**Option 2: Manual activation**
```bash
# Activate virtual environment first
source ../venv/bin/activate

# Then run Streamlit
streamlit run app.py
```

The app will automatically open in your browser at `http://localhost:8501`

## Model Details

- **Base Model**: `facebook/esm2_t12_35M_UR50D`
- **Strategy**: Fine-tuned last transformer layer + 2-layer MLP classifier
- **Input**: Protein sequences (amino acids)
- **Output**: Binary classification (0 = Toxic, 1 = Safe)
- **Best MCC**: 0.6001

## Troubleshooting

### Model file not found
- Make sure `best_model_state.bin` is in the same directory as `app.py` or `inference.py`
- Check file path in the script if needed

### Missing dependencies
- Install all requirements: `pip install torch transformers matplotlib seaborn pandas streamlit`
- Make sure you're in the virtual environment: `source ../venv/bin/activate`

### Streamlit command not found
- Activate the virtual environment first: `source ../venv/bin/activate`
- Or use the helper script: `./run_app.sh`

### CUDA errors
- The inference script automatically uses CPU if CUDA is not available
- CPU is fine for single-sequence inference
