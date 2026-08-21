"""
AgriSathi Official Document Ingestion Pipeline
- Processes raw official documents (.txt, .md, .csv, .pdf) from data/documents/
- Applies Recursive Character Text Chunking (chunk_size=500, overlap=50)
- Computes TF-IDF vector embeddings and metadata payloads
- Exports indexed vector store to data/embeddings/faiss_vector_store.json
"""

import os
import glob
import json
import re
import math
import time
from typing import List, Dict, Any

DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "../data/documents")
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "../data/embeddings")

def get_vector_store_path():
    candidates = [
        os.path.join(os.path.dirname(__file__), "faiss_vector_store.json"),
        os.path.join(os.path.dirname(__file__), "../data/embeddings/faiss_vector_store.json"),
        os.path.join(os.getcwd(), "backend/faiss_vector_store.json"),
        os.path.join(os.getcwd(), "data/embeddings/faiss_vector_store.json"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return candidates[0]

OUTPUT_VECTOR_STORE = get_vector_store_path()

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def extract_text_from_file(file_path: str) -> str:
    """Extracts raw text from .txt, .md, .csv, or .pdf files."""
    ext = os.path.splitext(file_path)[1].lower()
    
    if ext in [".txt", ".md", ".csv", ".json", ".log"]:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read().strip()
            
    elif ext == ".pdf":
        text = ""
        # Try standard pypdf text extraction
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception:
            pass

        if not text.strip():
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(file_path)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            except Exception:
                pass

        # If standard PDF text extraction returned minimal content (< 100 chars), perform OCR on scanned PDF pages
        if len(text.strip()) < 100:
            print(f"[PDF INGEST] Performing high-precision OCR on scanned PDF: {file_path}")
            try:
                import pypdfium2 as pdfium
                import numpy as np
                from rapidocr_onnxruntime import RapidOCR
                
                ocr_engine = RapidOCR()
                pdf = pdfium.PdfDocument(file_path)
                ocr_text = []

                for page_idx in range(len(pdf)):
                    img = np.array(pdf[page_idx].render(scale=1.5).to_pil())
                    result, _ = ocr_engine(img)
                    if result:
                        page_str = " ".join([res[1] for res in result if res[1]])
                        if page_str.strip():
                            ocr_text.append(page_str)
                
                if ocr_text:
                    return "\n\n".join(ocr_text).strip()
            except Exception as e:
                print(f"[PDF OCR ERROR] {e}")

        return text.strip()

    return ""


class DocumentIngestionPipeline:
    def __init__(self, docs_dir: str = DOCUMENTS_DIR):
        self.docs_dir = docs_dir
        self.chunks: List[Dict[str, Any]] = []

    def _tokenize(self, text: str) -> List[str]:
        """Multilingual tokenizer for text indexing."""
        text = text.lower()
        words = re.findall(r'\b[a-zA-Z0-9\u0900-\u097F]+\b', text)
        stop_words = {"ko", "ki", "ka", "ke", "mein", "par", "se", "hai", "hain", "aur", "ya", "bhi", "is", "us", "kya", "the", "a", "an", "in", "to", "for", "of", "and", "or", "is", "are"}
        return [w for w in words if len(w) > 1 and w not in stop_words]

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
                    # Split large paragraph by lines
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
        """Scans docs_dir, parses raw files, chunks text, and generates vector index."""
        print(f"[INGEST] Scanning raw document repository: {self.docs_dir}")
        os.makedirs(EMBEDDINGS_DIR, exist_ok=True)
        
        file_patterns = ["*.txt", "*.md", "*.csv", "*.pdf"]
        doc_files = []
        for pattern in file_patterns:
            doc_files.extend(glob.glob(os.path.join(self.docs_dir, pattern)))

        if not doc_files:
            print(f"[WARNING] No raw document files found in {self.docs_dir}. Flushing vector store...")
            vector_store_payload = {
                "ingest_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "total_files": 0,
                "total_chunks": 0,
                "chunk_size": CHUNK_SIZE,
                "chunk_overlap": CHUNK_OVERLAP,
                "chunks": []
            }
            with open(OUTPUT_VECTOR_STORE, "w", encoding="utf-8") as f:
                json.dump(vector_store_payload, f, indent=2, ensure_ascii=False)
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
                        "text": chunk_str,
                        "tokens": self._tokenize(chunk_str)
                    })

            except Exception as e:
                print(f"[ERROR] Failed to ingest {file_name}: {e}")

        # Compute TF-IDF Sparse Vector Embeddings
        total_chunks = max(1, len(all_chunks))
        df_counts: Dict[str, int] = {}
        for c in all_chunks:
            unique_t = set(c["tokens"])
            for t in unique_t:
                df_counts[t] = df_counts.get(t, 0) + 1

        for c in all_chunks:
            tf: Dict[str, int] = {}
            for t in c["tokens"]:
                tf[t] = tf.get(t, 0) + 1
            
            vector: Dict[str, float] = {}
            doc_len = max(1, len(c["tokens"]))
            for t, count in tf.items():
                idf = math.log((total_chunks + 1) / (df_counts.get(t, 0) + 1)) + 1
                vector[t] = round((count / doc_len) * idf, 4)
            
            c["vector"] = vector
            del c["tokens"]

        # Export to faiss_vector_store.json
        vector_store_payload = {
            "ingest_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "total_files": len(doc_files),
            "total_chunks": len(all_chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "chunks": all_chunks
        }

        with open(OUTPUT_VECTOR_STORE, "w", encoding="utf-8") as f:
            json.dump(vector_store_payload, f, indent=2, ensure_ascii=False)

        print(f"[OK] Successfully ingested {len(all_chunks)} chunks from {len(doc_files)} files into {OUTPUT_VECTOR_STORE}")
        return all_chunks

    def ingest_single_file(self, file_path: str) -> Dict[str, Any]:
        """Ingests a newly uploaded single file and updates the vector store database."""
        all_chunks = self.ingest()
        target_name = os.path.basename(file_path)
        added_chunks = [c for c in all_chunks if c.get("file_name") == target_name]
        return {
            "file_name": target_name,
            "chunks_added": len(added_chunks),
            "total_system_chunks": len(all_chunks)
        }

    def get_ingested_summary(self) -> List[Dict[str, Any]]:
        """Returns summary of all ingested document files."""
        if not os.path.exists(OUTPUT_VECTOR_STORE):
            return []

        try:
            with open(OUTPUT_VECTOR_STORE, "r", encoding="utf-8") as f:
                data = json.load(f)
                chunks = data.get("chunks", [])
                file_summary: Dict[str, Dict[str, Any]] = {}
                for c in chunks:
                    fname = c.get("file_name", "Unknown")
                    if fname not in file_summary:
                        file_path = os.path.join(self.docs_dir, fname)
                        size_kb = round(os.path.getsize(file_path) / 1024, 1) if os.path.exists(file_path) else 0.0
                        file_summary[fname] = {
                            "file_name": fname,
                            "title": c.get("title", fname),
                            "source": c.get("source", "Uploaded File"),
                            "chunks": 0,
                            "size_kb": size_kb,
                            "timestamp": data.get("ingest_timestamp", "")
                        }
                    file_summary[fname]["chunks"] += 1
                return list(file_summary.values())
        except Exception:
            return []


if __name__ == "__main__":
    pipeline = DocumentIngestionPipeline()
    pipeline.ingest()
