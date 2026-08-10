# AgriSathi — System Architecture

## End-to-End Pipeline

### 1. Data Layer
- **Raw Sources**: KCC CSV, Crop Recommendation CSV, Govt PDFs
- **Preprocessing**: clean_text() → dedup → Alpaca format
- **Storage**: CSV splits (train/val/test) + SQLite registry

### 2. Embedding & RAG Layer
- **Embedding Model**: BAAI/bge-small-en-v1.5 (384-dim)
- **Chunking**: RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
- **Vector Store**: FAISS (cosine similarity, top-k=3 retrieval)
- **Index**: Saved locally + backed up to Google Drive

### 3. Fine-tuning Layer
- **Base Model**: Mistral-7B-Instruct-v0.3
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **LoRA Config**: r=16, alpha=16, target all projection layers
- **Training**: SFTTrainer, 2 epochs, Alpaca prompt format

### 4. Inference Layer
- **RAG Flow**: Query → FAISS retrieve (top-3) → augment prompt → generate
- **Model**: Fine-tuned QLoRA adapter loaded with Unsloth
- **Output**: Hindi/Hinglish answer, 256 max new tokens

### 5. API Layer
- **Framework**: FastAPI
- **Endpoints**: POST /query, GET /metrics, GET /health
- **Logging**: SQLite query_logs table

### 6. Frontend Layer
- **Dashboard**: Single-file HTML (Chart.js)
- **Sections**: Overview, Chat Demo, Metrics, Dataset, Error Analysis, Architecture
- **Mobile**: Flutter App (planned)

## Why These Technology Choices?

| Choice | Reason |
|--------|--------|
| Mistral-7B | Strong multilingual, instruction-tuned, fits T4 GPU |
| QLoRA | Only 0.57% params trainable, fits 15GB VRAM |
| Unsloth | 2-3x faster training than vanilla PEFT |
| FAISS | CPU-compatible, fast similarity search, no server needed |
| BGE-small | Best small embedding for retrieval tasks |
| FastAPI | Async, auto-docs, type-safe |
| SQLite | Zero-config, good for logging/registry |
