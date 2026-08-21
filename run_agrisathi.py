"""
AgriSathi AI Platform — Single-Click Startup Launcher
1. Verifies raw document vector ingestion database
2. Starts FastAPI Uvicorn server on http://localhost:8000
3. Opens AgriSathi Developer Dashboard in default browser
"""

import os
import sys
import time
import subprocess
import webbrowser

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("=" * 70)
    print("🌾 AgriSathi AI Platform — Single-Click Launcher")
    print("=" * 70)

    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(project_root, "backend"))

    # 1. Ingestion Check
    vector_store = os.path.join(project_root, "data", "embeddings", "faiss_vector_store.json")
    if not os.path.exists(vector_store):
        print("\n[INGEST] Vector store missing. Executing Document Ingestion Pipeline...")
        from ingest_documents import DocumentIngestionPipeline
        DocumentIngestionPipeline().ingest()
    else:
        print("\n[OK] Raw document vector database loaded successfully.")

    # 2. Launch Uvicorn FastAPI Server
    print("\n[SERVER] Starting FastAPI Backend on http://localhost:8000 ...")
    backend_dir = os.path.join(project_root, "backend")
    
    server_process = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
        cwd=backend_dir
    )

    # Give server 2 seconds to initialize
    time.sleep(2.5)

    # 3. Open Frontend Dashboard in Default Web Browser
    dashboard_path = os.path.join(project_root, "frontend", "dashboard", "index.html")
    dashboard_url = "file:///" + dashboard_path.replace("\\", "/")
    
    print(f"\n[BROWSER] Opening AgriSathi Dashboard at: {dashboard_url}")
    webbrowser.open(dashboard_url)

    print("\n" + "=" * 70)
    print("🚀 AgriSathi AI Backend is LIVE on http://localhost:8000")
    print("📄 Interactive API Docs: http://localhost:8000/docs")
    print("Press Ctrl+C in this terminal to stop the server.")
    print("=" * 70)

    try:
        server_process.wait()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down AgriSathi AI server...")
        server_process.terminate()

if __name__ == "__main__":
    main()
