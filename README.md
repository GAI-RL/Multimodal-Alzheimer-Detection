# Multimodal Alzheimer's Detection

Detects dementia from speech by combining **text** (DistilBERT) and **audio** (Wav2Vec2) embeddings with a cross-attention fusion model and a small classifier head. Data: DementiaBank Pitt (Control vs Dementia).

## Repository structure
| Path | Description |
|------|-------------|
| `notebooks/trainingandevaluation_code.ipynb` | Feature extraction, training and evaluation |
| `models/best_model.pth` | Best checkpoint (model + optimizer state) |
| `models/95_accuracy_trainedmodel.pth` | Trained model weights (state dict) |
| `features/multimodal_features.pkl` | Pre-extracted features (768 text + 768 audio = 1536 per sample) |
| `docs/` | Dataset documentation |

Large files (`.pth`, `.pkl`) are stored with Git LFS.

## Setup
```bash
git lfs install
git clone https://github.com/GAI-RL/Multimodal-Alzheimer-Detection.git
cd Multimodal-Alzheimer-Detection
pip install -r requirements.txt
```

## Usage
Open `notebooks/trainingandevaluation_code.ipynb` (written for Google Colab; update the Google Drive paths if you run it locally).

## Notes
- Pretrained backbones: `distilbert-base-uncased` and `facebook/wav2vec2-base-960h` (downloaded automatically from Hugging Face).
- Results and evaluation details are in the notebook.
- The pickle file should only be loaded from trusted sources.
