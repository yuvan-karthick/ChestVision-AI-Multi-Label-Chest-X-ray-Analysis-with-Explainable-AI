# AI-Assisted Multi-Label Chest X-ray Analysis

### NIH ChestX-ray14 • CNNs • Grad-CAM • RAG • Gemini • Streamlit

An academic medical-AI project for multi-label chest X-ray analysis using deep learning, explainability, Retrieval-Augmented Generation (RAG), and an LLM interpretation layer.

> **Disclaimer:** This is an academic research prototype, not a medical diagnostic device. Model predictions are not definitive diagnoses, and Grad-CAM is not clinical proof.

---

## Overview

The project uses the NIH ChestX-ray14 dataset to predict 14 thoracic findings from chest X-rays.

### Pipeline

```text
Chest X-ray
    ↓
Preprocessing
    ↓
VGG-19
    ↓
14-label predictions
    ↓
Grad-CAM
    ↓
RAG retrieval
    ↓
Gemini
    ↓
Grounded explanation
    ↓
Streamlit
```

The project combines:

- Custom CNN
- ResNet-18
- VGG-19
- Grad-CAM
- FAISS-based RAG
- Sentence Transformers
- Gemini 3.1 Flash-Lite
- Streamlit

---

## Dataset

**NIH ChestX-ray14**

The task is multi-label classification with 14 independent outputs:

```text
Atelectasis
Cardiomegaly
Effusion
Infiltration
Mass
Nodule
Pneumonia
Pneumothorax
Consolidation
Edema
Emphysema
Fibrosis
Pleural_Thickening
Hernia
```

`No Finding` is represented as an all-zero 14-label target rather than a separate class.

### Dataset split

| Split | Images | Patients |
|---|---:|---:|
| Train | 78,566 | 21,563 |
| Validation | 16,106 | 4,621 |
| Test | 17,448 | 4,621 |

Patient-wise splitting was used to reduce data leakage.

---

## Preprocessing

- Original images: 1024×1024 grayscale
- Resized to 224×224
- Offline image cache
- ImageNet normalization
- Grayscale adaptation for pretrained CNNs

Class imbalance was handled using weighted `BCEWithLogitsLoss`.

---

## Model Results

Three models were trained:

| Model | Test Macro/Mean AUROC |
|---|---:|
| Custom CNN | 0.7697 |
| ResNet-18 | 0.7841 |
| VGG-19 | ~0.7989 |

VGG-19 is used for the final application.

### VGG-19

```text
Grayscale input
↓
VGG-19 backbone
↓
Adaptive Average Pooling
↓
512 → 256
↓
Dropout(0.3)
↓
256 → 14
```

---

## Grad-CAM

Grad-CAM is used to visualize regions that contributed to model predictions.

Target layers:

```text
Custom CNN  → features[12]
ResNet-18   → layer4[1].conv2
VGG-19      → features[34]
```

Grad-CAM represents **model attribution**, not proof of a lesion or diagnosis.

---

## RAG

The RAG system retrieves curated medical and dataset information before generating the initial explanation.

### RAG V2

```text
56 semantic chunks
all-MiniLM-L6-v2
384-dimensional normalized embeddings
FAISS IndexFlatIP
```

Sources include:

- NIH / NCBI
- MedlinePlus
- RadiologyInfo
- NHLBI
- NCBI MedGen
- ACR / RSNA sources

The corpus includes finding-specific information for the 14 model labels.

---

## Gemini

The application uses:

```text
Gemini 3.1 Flash-Lite
```

For the initial analysis:

```text
CNN prediction
+ Grad-CAM information
+ retrieved RAG evidence
→ Gemini
```

Gemini generates:

- Medical context
- Interpretation
- Limitations
- Citation references

The LLM does not change the CNN prediction.

### Follow-up questions

After the initial analysis, follow-up questions are handled directly by Gemini using the initial result as context.

---

## Streamlit Application

The application provides:

- X-ray upload
- 14-label predictions
- Top prediction
- Grad-CAM
- Medical context
- AI interpretation
- Evidence status
- Sources
- Limitations
- Follow-up chat

---

## Project Structure

```text
Chest_Xray_AI/
│
├── app.py
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── Chest_Xray_AI.ipynb
│   ├── RAG.ipynb
│   └── LLM.ipynb
│
├── data/
├── checkpoints/
├── results/
│
└── rag/
    └── v2/
        ├── documents/
        ├── chunks/
        ├── metadata/
        └── vector_store/
```

Large datasets, model checkpoints, generated RAG files, and secrets are excluded from Git.

---

## Installation

```bash
git clone <YOUR_REPOSITORY_URL>
cd Chest_Xray_AI

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
pip install streamlit google-genai
```

The project was developed with Python 3.11 and PyTorch 2.11.

---

## Gemini API Key

Set the API key as an environment variable.

PowerShell:

```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY"
```

The API key should never be hardcoded or committed to GitHub.

---

## Run

```powershell
python -m streamlit run app.py
```

Then open the local Streamlit URL in your browser.

---

## Limitations

- ChestX-ray14 labels are research annotations, not definitive diagnoses.
- Model probabilities are not clinical probabilities or diagnoses.
- Grad-CAM is an attribution method, not lesion localization.
- RAG quality depends on corpus coverage and retrieval.
- LLM-generated information may contain errors.
- The system has not undergone clinical validation or external validation.

---

## References

- NIH ChestX-ray14: https://pmc.ncbi.nlm.nih.gov/articles/PMC9131331/
- RadiologyInfo Chest X-ray: https://www.radiologyinfo.org/en/info/chestrad
- RadiologyInfo Chest X-ray Report: https://www.radiologyinfo.org/en/info/article-chest-xray-report

---

## Author

**Yuvan Karthick**

AI/ML • Computer Vision • Healthcare AI • RAG • LLM Applications
