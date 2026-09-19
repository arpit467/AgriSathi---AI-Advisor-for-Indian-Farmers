"""
AgriSathi RAG Pipeline
ChromaDB Vector Store + Fine-tuned LLM Integration
"""

import os
import torch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
CHUNK_SIZE      = 512
CHUNK_OVERLAP   = 50
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "../data/chroma_db_pipeline")


class AgriSathiRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.model       = None
        self.tokenizer   = None

    def load_embeddings(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True},
        )
        print(f"✅ Embeddings loaded on {device}")

    def build_vector_store(self, documents: list[str]):
        """Build ChromaDB vector store from raw text documents."""
        if self.embeddings is None:
            self.load_embeddings()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        docs = [Document(page_content=d) for d in documents]
        chunks = splitter.split_documents(docs)

        os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
        self.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=self.embeddings,
            persist_directory=CHROMA_PERSIST_DIR,
            collection_name="agrisathi_pipeline"
        )
        print(f"✅ ChromaDB vector store built: {len(chunks)} chunks → {CHROMA_PERSIST_DIR}")

    def load_vector_store(self):
        if self.embeddings is None:
            self.load_embeddings()
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings,
            collection_name="agrisathi_pipeline"
        )
        print("✅ ChromaDB vector store loaded")

    def retrieve(self, query: str, k: int = 3) -> list[str]:
        """Retrieve top-k relevant chunks for a query."""
        if self.vectorstore is None:
            self.load_vector_store()
        results = self.vectorstore.similarity_search(query, k=k)
        return [r.page_content for r in results]

    def load_finetuned_model(self, model_path: str = None):
        """Load fine-tuned QLoRA model for inference."""
        from unsloth import FastLanguageModel

        model_path = model_path or os.path.join(
            os.path.dirname(__file__), "../models/agrisathi-finetuned"
        )
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_path,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(self.model)
        print("✅ Fine-tuned model loaded")

    def generate(self, question: str, use_rag: bool = True, max_new_tokens: int = 256) -> dict:
        """Full RAG pipeline: retrieve → augment → generate."""
        import time

        # Step 1: Retrieve context
        context_chunks = self.retrieve(question) if use_rag else []
        context = "\n".join(context_chunks)

        # Step 2: Build prompt
        prompt = f"""<s>[INST] Aap AgriSathi AI ho — ek farming expert jo kisano ki madad karta hai.
Niche diye gaye context ka use karke sawaal ka jawab do.

Context:
{context}

Sawaal: {question} [/INST]"""

        # Step 3: Generate
        start = time.time()
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        elapsed = round(time.time() - start, 2)

        answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = answer.split("[/INST]")[-1].strip()

        return {
            "answer": answer,
            "retrieved_chunks": context_chunks,
            "inference_time": elapsed,
        }


# Singleton instance
rag_pipeline = AgriSathiRAG()
