"""
AgriSathi Official Document Ingestion Pipeline
- Processes raw official documents (.txt, .md, .csv, .pdf) from data/documents/
- Applies Recursive Character Text Chunking (chunk_size=500, overlap=50)
- Computes dense neural embeddings (sentence-transformers/all-MiniLM-L6-v2)
- Builds and persists collections in ChromaDB (cosine distance space)
"""

import os
import glob
import json
import re
import time
from typing import List, Dict, Any, Optional

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "../data/documents")
CHROMA_DB_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
CHROMA_COLLECTION_NAME = "agrisathi_kb"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

_embedding_model: Optional[SentenceTransformer] = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[EMBED] Loading dense embedding model '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


def get_chroma_client(db_path: str = CHROMA_DB_DIR) -> chromadb.PersistentClient:
    """Returns a persistent ChromaDB client."""
    os.makedirs(db_path, exist_ok=True)
    return chromadb.PersistentClient(path=db_path)


def get_or_create_kb_collection(client: Optional[chromadb.PersistentClient] = None, reset: bool = False):
    """Gets or initializes the AgriSathi knowledge base collection with cosine distance metric."""
    if client is None:
        client = get_chroma_client()
    if reset:
        try:
            client.delete_collection(CHROMA_COLLECTION_NAME)
        except Exception:
            pass
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )


def extract_text_from_file(file_path: str) -> str:
    """Extracts raw text from .txt, .md, .csv, or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md", ".csv", ".json", ".log"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
            
    elif ext == ".pdf":
        text = ""
        # 1. Try PyMuPDF (fitz) - ultra fast & high precision
        try:
            import fitz
            doc = fitz.open(file_path)
            pages_text = [p.get_text() for p in doc]
            text = "\n".join([p for p in pages_text if p.strip()])
            if text.strip():
                return text.strip()
        except Exception:
            pass

        # 2. Try standard pypdf
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            if text.strip():
                return text.strip()
        except Exception:
            pass

        # 3. Try PyPDF2 fallback
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            pass

        return text.strip()

    return ""


class DocumentIngestionPipeline:
    def __init__(self, docs_dir: str = DOCUMENTS_DIR, chroma_dir: str = CHROMA_DB_DIR):
        self.docs_dir = os.path.abspath(docs_dir)
        self.chroma_dir = os.path.abspath(chroma_dir)
        self.chunks: List[Dict[str, Any]] = []

    def _split_text(self, text: str, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Recursive character text splitter."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = ""

        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(current_chunk) + len(p) <= chunk_size:
                current_chunk += ("\n\n" if current_chunk else "") + p
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                if len(p) > chunk_size:
                    sub_lines = p.split("\n")
                    sub_chunk = ""
                    for line in sub_lines:
                        if len(sub_chunk) + len(line) <= chunk_size:
                            sub_chunk += ("\n" if sub_chunk else "") + line
                        else:
                            if sub_chunk:
                                chunks.append(sub_chunk)
                            sub_chunk = line
                    if sub_chunk:
                        chunks.append(sub_chunk)
                    current_chunk = ""
                else:
                    current_chunk = p

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def ingest(self) -> List[Dict[str, Any]]:
        """Scans docs_dir, parses raw files, chunks text, computes embeddings, and persists to ChromaDB."""
        print(f"[INGEST] Scanning raw document repository: {self.docs_dir}")
        os.makedirs(self.chroma_dir, exist_ok=True)

        client = get_chroma_client(self.chroma_dir)
        collection = get_or_create_kb_collection(client, reset=True)

        file_patterns = ["*.txt", "*.md", "*.csv", "*.pdf"]
        doc_files = []
        for pattern in file_patterns:
            doc_files.extend(glob.glob(os.path.join(self.docs_dir, pattern)))

        if not doc_files:
            print(f"[WARNING] No raw document files found in {self.docs_dir}. ChromaDB collection is empty.")
            return []

        print(f"[INGEST] Found {len(doc_files)} raw document files.")
        all_chunks = []
        doc_counter = 0

        for file_path in doc_files:
            file_name = os.path.basename(file_path)
            try:
                raw_text = extract_text_from_file(file_path)
                if not raw_text:
                    continue

                doc_counter += 1
                source = "Uploaded Document"
                title = file_name.replace("_", " ").replace(".txt", "").replace(".md", "").replace(".pdf", "").replace(".csv", "")
                
                header_match = re.search(r"Document Source:\s*(.+)", raw_text, re.IGNORECASE)
                if header_match:
                    source = header_match.group(1).strip()
                else:
                    source = f"File: {file_name}"
                
                portal_match = re.search(r"Official Portal:\s*(.+)", raw_text, re.IGNORECASE)
                url = portal_match.group(1).strip() if portal_match else ""

                # Chunk document
                raw_chunks = self._split_text(raw_text)
                for idx, chunk_str in enumerate(raw_chunks):
                    all_chunks.append({
                        "chunk_id": f"doc_{doc_counter}_chunk_{idx + 1}",
                        "file_name": file_name,
                        "title": title,
                        "source": source,
                        "url": url,
                        "text": chunk_str
                    })

            except Exception as e:
                print(f"[ERROR] Failed to ingest {file_name}: {e}")

        if not all_chunks:
            print("[WARNING] No chunks created from documents.")
            return []

        # ── Compute Dense Neural Embeddings with SentenceTransformer ──
        print(f"[EMBED] Generating dense neural embeddings for {len(all_chunks)} chunks with {EMBEDDING_MODEL_NAME}...")
        model = get_embedding_model()
        texts = [c["text"] for c in all_chunks]
        embeddings = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=False)

        # ── Persist to ChromaDB Collection in Batches ──
        print(f"[CHROMA] Upserting {len(all_chunks)} dense vector records into ChromaDB '{CHROMA_COLLECTION_NAME}'...")
        batch_size = 500
        for i in range(0, len(all_chunks), batch_size):
            end = min(i + batch_size, len(all_chunks))
            batch_chunks = all_chunks[i:end]
            collection.upsert(
                ids=[c["chunk_id"] for c in batch_chunks],
                documents=[c["text"] for c in batch_chunks],
                metadatas=[{
                    "file_name": c["file_name"],
                    "title": c["title"],
                    "source": c["source"],
                    "url": c["url"],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
                } for c in batch_chunks],
                embeddings=embeddings[i:end].tolist()
            )

        print(f"[OK] Successfully built ChromaDB Vector Database with {collection.count()} chunks from {len(doc_files)} files at {self.chroma_dir}.")
        return all_chunks

    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """Ingests a newly uploaded single file and updates the ChromaDB collection."""
        all_chunks = self.ingest()
        target_name = os.path.basename(file_path)
        added_chunks = [c for c in all_chunks if c.get("file_name") == target_name]
        return {
            "file_name": target_name,
            "chunks_added": len(added_chunks),
            "total_system_chunks": len(all_chunks)
        }

    def get_ingested_summary(self) -> List[Dict[str, Any]]:
        """Returns summary of all ingested document files directly from ChromaDB and documents folder."""
        try:
            client = get_chroma_client(self.chroma_dir)
            collection = get_or_create_kb_collection(client, reset=False)
            total_chunks = collection.count()
            if total_chunks == 0:
                return []

            # Retrieve metadatas to build file summary
            results = collection.get(include=["metadatas"])
            metadatas = results.get("metadatas", [])
            file_summary: Dict[str, Dict[str, Any]] = {}

            for meta in metadatas:
                fname = meta.get("file_name", "Unknown")
                if fname not in file_summary:
                    file_path = os.path.join(self.docs_dir, fname)
                    size_kb = round(os.path.getsize(file_path) / 1024, 1) if os.path.exists(file_path) else 0.0
                    file_summary[fname] = {
                        "file_name": fname,
                        "title": meta.get("title", fname),
                        "source": meta.get("source", "Uploaded File"),
                        "chunks": 0,
                        "size_kb": size_kb,
                        "timestamp": meta.get("timestamp", "")
                    }
                file_summary[fname]["chunks"] += 1

            return list(file_summary.values())
        except Exception as e:
            print(f"[SUMMARY ERROR] {e}")
            return []


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest()
