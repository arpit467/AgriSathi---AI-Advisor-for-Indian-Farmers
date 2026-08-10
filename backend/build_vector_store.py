"""
AgriSathi FAISS Vector Store Builder
Extracts text from datasets and saves FAISS vector index.
"""

import os
import pandas as pd
import sqlite3

def build_vector_store():
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "../data/processed")
    index_dir = os.path.join(base_dir, "../data/embeddings/faiss_index")
    os.makedirs(index_dir, exist_ok=True)

    documents = [
        "Gehu mein pila pan zinc ya nitrogen ki kami se hota hai. Zinc sulphate 25 kg/hectare daalo.",
        "PM-KISAN yojana mein har saal 6000 rupaye teen kiston mein milte hain (2000 rupaye har 4 mahine).",
        "Chawal mein blast disease Magnaporthe oryzae fungus se hoti hai. Tricyclazole 75% WP spray karo.",
        "Drip irrigation mein 40-50% paani bachta hai. Sabzi aur baagan ke liye best hai.",
        "Urea mein 46% nitrogen hota hai — yeh sabse zyada concentrated nitrogenous fertilizer hai.",
        "Tamatar ke liye NPK 19:19:19 aur calcium nitrate best fertilizer hai. Excessive nitrogen se bachne ke liye urea kam daalen.",
        "Soil health card yojana ke tehat mitti ka pH test 3 saal mein ek baar har kisan ke khet ke liye hota hai.",
        "Kisan Credit Card (KCC) ke zariye 3 lakh tak ka loan 4% subsidized interest rate par milta hai."
    ]

    # Load from train.csv if available
    train_csv = os.path.join(data_dir, "train.csv")
    if os.path.exists(train_csv):
        try:
            df = pd.read_csv(train_csv)
            for idx, row in df.iterrows():
                instr = str(row.get('instruction', ''))
                output = str(row.get('output', ''))
                if instr and output:
                    documents.append(f"Q: {instr} | A: {output}")
        except Exception as e:
            print(f"Notice loading train.csv: {e}")

    try:
        from langchain_community.vectorstores import FAISS
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document

        print(f"Building FAISS index with {len(documents)} documents...")
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        docs = [Document(page_content=d) for d in documents]
        vectorstore = FAISS.from_documents(docs, embeddings)
        vectorstore.save_local(index_dir)
        print(f"[OK] FAISS vector store saved to {index_dir}")
    except Exception as e:
        print(f"[NOTICE] Vector store builder info: {e}")
        # Save a text index manifest if HuggingFace/FAISS model download fails locally
        manifest_path = os.path.join(index_dir, "manifest.json")
        import json
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump({"total_documents": len(documents), "sample": documents[:3]}, f, indent=2)
        print(f"[OK] Created text index manifest fallback at {manifest_path}")

if __name__ == "__main__":
    build_vector_store()
