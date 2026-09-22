import os
import sys
import json
import glob
import subprocess

def check_dependencies():
    """Checks if faiss and numpy are installed, and installs them if missing."""
    try:
        import faiss
        import numpy as np
        print("✅ Dependencies found.")
    except ImportError:
        print("🔍 Dependency missing. Attempting to install 'faiss-cpu' and 'numpy'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "faiss-cpu", "numpy"])
            print("🚀 Dependencies installed successfully. Restarting imports...")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please run: pip install faiss-cpu numpy")
            sys.exit(1)

# Check dependencies before importing heavy libraries
check_dependencies()

import faiss
import numpy as np

# -------- Configuration --------
EMBEDDINGS_DIR = "embeddings"
FAISS_DB_DIR = "faiss_db"

# Ensure output directory exists
if not os.path.exists(FAISS_DB_DIR):
    os.makedirs(FAISS_DB_DIR)

def main():
    # Find all JSON chunk files in the embeddings folder
    embedding_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.json"))
    
    if not embedding_files:
        print(f"❌ No JSON files found in the '{EMBEDDINGS_DIR}' directory. Did you run embedder.py?")
        return
        
    print(f"🔎 Found {len(embedding_files)} embedding file(s) to process.")
    
    all_embeddings = []
    metadata = []
    
    for file_path in embedding_files:
        print(f"⏳ Processing {file_path}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                chunks = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
                
        source_file = os.path.basename(file_path).replace(".json", ".pdf")
        
        for chunk in chunks:
            # Check if embedding exists
            if "embedding" not in chunk:
                print(f"⚠️ Warning: Chunk missing 'embedding' in {file_path}. Skipping.")
                continue
                
            embedding = chunk.get("embedding")
            text = chunk.get("text", "")
            context = chunk.get("context", "")
            
            # Store the underlying vectors
            all_embeddings.append(embedding)
            
            # Store the metadata so we know exactly payload corresponds to Vector Index ID
            metadata.append({
                "source": source_file,
                "text": text,
                "context": context
            })
            
    if not all_embeddings:
        print("❌ No valid embeddings found in any of the files.")
        return

    print("⚡ Converting vectors to Fast NumPy Matrix...")
    # FAISS expects float32
    embedding_matrix = np.array(all_embeddings, dtype=np.float32)
    dimension = embedding_matrix.shape[1]
    
    print(f"📊 Vector statistics: {embedding_matrix.shape[0]} documents, {dimension} dimensions.")
    
    # Using Inner Product since sentence-transformers outputs normalized embeddings.
    # Inner Product of normalized vectors is identical to Cosine Similarity.
    print("🏗️ Building FAISS IndexFlatIP (Cosine Similarity)...")
    index = faiss.IndexFlatIP(dimension)
    
    # Optional: ensure we are feeding them properly (all-MiniLM-L6-v2 normalizes by default, but to be sure we can re-normalize)
    faiss.normalize_L2(embedding_matrix)
    
    # Add vectors to FAISS
    index.add(embedding_matrix)
    print(f"✅ Loaded {index.ntotal} vectors into FAISS database.")
    
    # Save the FAISS Index
    faiss_index_path = os.path.join(FAISS_DB_DIR, "vector_store.index")
    print(f"💾 Saving binary FAISS index to {faiss_index_path}...")
    faiss.write_index(index, faiss_index_path)
    
    # Save the parallel Metadata map
    faiss_metadata_path = os.path.join(FAISS_DB_DIR, "metadata.json")
    print(f"💾 Saving Parallel Metadata Dictionary to {faiss_metadata_path}...")
    with open(faiss_metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4, ensure_ascii=False)
        
    print(f"\n🚀 ALL DONE! Query away.")

if __name__ == "__main__":
    main()
