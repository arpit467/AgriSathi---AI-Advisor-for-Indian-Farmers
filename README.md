# 🌾 AgriSathi AI Advisor
### Domain-Specific RAG + Fine-tuned LLM for Indian Farmers

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)](https://fastapi.tiangolo.com)
[![Model](https://img.shields.io/badge/Model-Mistral--7B%20QLoRA-orange)](https://huggingface.co/mistralai)
[![License](https://img.shields.io/badge/License-MIT-lightgrey)](LICENSE)

---

## 📌 Problem Statement
Indian farmers face three critical problems:
1. Don't understand complex government schemes
2. Get unreliable advice from generic LLMs (hallucinations)
3. Crop disease confusion without localized guidance

## 💡 Solution
AgriSathi — a fine-tuned AI assistant that:
- Answers farming queries in **Hindi/Hinglish**
- Gives **region-aware** advice via RAG
- **Reduces hallucination** using FAISS retrieval
- Explains government schemes, diseases, irrigation, fertilizers

---

## ✅ Evaluation Criteria Coverage

| # | Criterion | Implementation |
|---|-----------|---------------|
| i | Dataset quality + preprocessing + split | KCC + Crop + Govt data, 80/10/10 split, Alpaca format |
| ii | PEFT (QLoRA) fine-tuning + justification | Mistral-7B, LoRA r=16, Unsloth, 4-bit NF4 |
| iii | Baseline comparison (3 models) | Base / Prompt-Engineered / Fine-tuned |
| iv | Data storage | FAISS vector DB + SQLite (3 tables) |
| v | BLEU, ROUGE-1/2/L evaluation | 50-sample test set evaluation |
| vi | Hallucination + error analysis | 5 known-answer cases, 3 error categories |
| vii | Real-world applicability | Hindi/Hinglish, farmer-relevant topics |
| + | Frontend UI | Dashboard + Chat UI + Architecture view |

---

## 📁 Project Structure

```
AgriSathi/
├── notebooks/
│   └── AgriSathi_Complete.ipynb   ← Main Colab notebook (all 7 criteria)
├── backend/
│   ├── main.py                    ← FastAPI REST API
│   ├── rag_pipeline.py            ← FAISS RAG + LLM pipeline
│   ├── evaluate.py                ← Full evaluation suite
│   └── requirements.txt
├── frontend/
│   └── dashboard/
│       └── index.html             ← Dashboard UI (open in browser)
├── data/
│   ├── raw/                       ← Downloaded datasets
│   ├── processed/
│   │   ├── train.csv              ← 80% split
│   │   ├── val.csv                ← 10% split
│   │   └── test.csv               ← 10% split
│   └── embeddings/
│       └── faiss_index/           ← FAISS vector store
├── models/
│   └── agrisathi-finetuned/       ← Saved QLoRA model
├── results/
│   ├── model_comparison.csv       ← 3-way metric comparison
│   └── hallucination_analysis.json
└── docs/
    └── architecture.md
```

---

## ⚙️ System Architecture

```
User (Flutter App / Dashboard)
        ↓
  FastAPI Backend (main.py)
        ↓
  RAG Pipeline (rag_pipeline.py)
        ↓
  FAISS Vector DB ←── BAAI/bge-small-en-v1.5
        ↓
  Fine-tuned LLM (Mistral-7B QLoRA)
        ↓
  Knowledge Base (KCC + Crop Data + Govt PDFs)
        ↓
  SQLite (query_logs, training_runs, dataset_registry)
```

---

## 🚀 Quick Start

### 1. Run Dashboard (No setup needed)
```bash
# Just open in browser:
open frontend/dashboard/index.html
```

### 2. Run Backend API
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
# API docs: http://localhost:8000/docs
```

### 3. Run Evaluation
```bash
cd backend
python evaluate.py
```

### 4. Train on Google Colab
```
1. Open notebooks/AgriSathi_Complete.ipynb in Google Colab
2. Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Download model from Drive → put in models/agrisathi-finetuned/
```

---

## 📊 Results

| Model | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|-------|--------|---------|---------|---------|
| Base Mistral-7B | 0.112 | 0.284 | 0.118 | 0.201 |
| Prompt-Engineered | 0.187 | 0.341 | 0.164 | 0.263 |
| **AgriSathi QLoRA** | **0.341** | **0.512** | **0.298** | **0.421** |

**Improvement vs Base: +204% BLEU | +109% ROUGE-L**

---

## 🧠 Model Details

- **Base**: `mistralai/Mistral-7B-Instruct-v0.3`
- **Fine-tuning**: QLoRA (4-bit NF4, LoRA r=16, alpha=16)
- **Trainable params**: ~41M / 7.24B (0.57%)
- **Training**: 2 epochs, lr=2e-4, adamw_8bit
- **Framework**: Unsloth + HuggingFace TRL

## 📚 Dataset

| Source | Samples | Language |
|--------|---------|----------|
| Kisan Call Center (KCC) | 8,500 | Hinglishi/Hindi |
| Crop Recommendation | 2,200 | English→Hinglish |
| Govt Scheme PDFs | 1,800 | Hindi |
| **Total** | **12,500** | Multi |

---

## 👥 Team
**AgriSathi Team** — GenAI Project 2025
