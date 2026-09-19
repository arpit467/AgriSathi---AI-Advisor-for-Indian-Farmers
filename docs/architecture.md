# AgriSathi — System Architecture

## End-to-End Pipeline

### 1. Data Layer
- **Raw Sources**: KCC CSV, Crop Recommendation CSV, Govt PDFs
- **Preprocessing**: clean_text() → dedup → Alpaca format
- **Storage**: CSV splits (train/val/test) + SQLite registry

### 2. Embedding & RAG Layer
- **Embedding Model**: all-MiniLM-L6-v2 / BAAI/bge-small-en-v1.5 (384-dim)
- **Chunking**: RecursiveCharacterTextSplitter (500 chars, 50 overlap)
- **Vector Store**: ChromaDB (cosine similarity, persistent collection)
- **Index**: Persistent local SQLite + HNSW storage in `data/chroma_db`

### 3. Fine-tuning Layer
- **Base Model**: Mistral-7B-Instruct-v0.3
- **Method**: QLoRA (4-bit quantization + LoRA adapters)
- **LoRA Config**: r=16, alpha=16, target all projection layers
- **Training**: SFTTrainer, 2 epochs, Alpaca prompt format

### 4. Inference Layer
- **RAG Flow**: Query → ChromaDB retrieve (top-k) → augment prompt → generate
- **Model**: Fine-tuned QLoRA adapter loaded with Unsloth / Groq API Fallback
- **Output**: Hindi/Hinglish answer, 256 max new tokens

### 5. API Layer
- **Framework**: FastAPI
- **Endpoints**: POST /query, GET /metrics, GET /health, POST /inspector/test
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
| ChromaDB | Embedded persistent vector DB, native metadata filtering & CRUD |
| BGE-small | Best small embedding for retrieval tasks |
| FastAPI | Async, auto-docs, type-safe |
| SQLite | Zero-config, good for logging/registry |
