import sys
import os

# Add root directory and backend directory to Python sys.path for Vercel
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'backend')))

from backend.main import app

# Export ASGI app for Vercel Serverless Python Engine
__all__ = ["app"]
