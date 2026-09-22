import os
import sys
import json
import subprocess

def check_dependencies():
    """Checks required dependencies for retrieval."""
    try:
        import faiss
        import numpy as np
        import sentence_transformers
        import torch
    except ImportError:
        print("🔍 Dependencies missing. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu", "numpy", "sentence-transformers", "torch"])
        print("🚀 Installed successfully. Restarting imports...")

check_dependencies()

import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# -------- Configuration --------
FAISS_DB_DIR = "faiss_db"
INDEX_PATH = os.path.join(FAISS_DB_DIR, "vector_store.index")
METADATA_PATH = os.path.join(FAISS_DB_DIR, "metadata.json")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_database():
    """Loads the FAISS index and the parallel metadata."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("❌ FAISS database not found! Please run 'python build_faiss.py' first.")
        sys.exit(1)
        
    print(f"📥 Loading FAISS index from {INDEX_PATH}...")
    index = faiss.read_index(INDEX_PATH)
    
    print(f"📥 Loading metadata from {METADATA_PATH}...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return index, metadata

def main():
    print("="*60)
    print("    RAG Retrieval Engine Initializing...")
    print("="*60)
    
    # 1. Load the database
    index, metadata = load_database()
    print(f"📊 Database ready. Total vectors loaded: {index.ntotal}")
    
    if index.ntotal == 0:
        print("❌ The FAISS index is empty.")
        return
        
    # 2. Load the embedding model
    print(f"\n🧠 Loading model: {MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(MODEL_NAME, device=device)
    print(f"✅ Model loaded on {device}.")
    
    # 3. Interactive Search Loop
    while True:
        print("\n" + "="*60)
        query = input("🔎 Enter your search query (or 'quit' to exit): ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        # --- Perform the Search ---
        # A. Embed the text query into a vector representation
        # all-MiniLM-L6-v2 normalizes unit length naturally, but convert_to_numpy is required
        query_embedding = model.encode([query], convert_to_numpy=True)
        
        # B. Make sure it's float32 (FAISS requirement)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # C. Search the FAISS Index (k=3 for Top-3 results)
        k = 3
        # D stores the distances/similarities, I stores the integer IDs
        D, I = index.search(query_embedding, k)
        
        print("\n🎯 TOP 3 RETRIEVED CHUNKS:")
        print("-" * 60)
        
        # D[0] and I[0] contain lists corresponding to the 1st query in our batch
        for rank in range(k):
            vector_id = int(I[0][rank])
            similarity = D[0][rank]
            
            # FAISS returns -1 if there are fewer vectors in the database than 'k'
            if vector_id == -1:
                continue
                
            # D. Map Vector ID back to the Human-Readable Document details
            chunk_data = metadata[vector_id]
            source = chunk_data.get("source", "Unknown PDF")
            context = chunk_data.get("context", "None")
            text = chunk_data.get("text", "")
            
            # Print the extracted RAG Information
            print(f"[{rank+1}] SCORE: {similarity:.4f}  |  SOURCE: {source}")
            if context:
                print(f"    CONTEXT: {context}")
            print(f"    TEXT: {text[:300]}...\n") # snippet of 300 chars to save terminal space

if __name__ == "__main__":
    main()
