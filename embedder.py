import os
import sys
import json
import glob
import subprocess

def check_dependencies():
    """Checks if sentence-transformers is installed, and installs it if missing."""
    try:
        import sentence_transformers
        import torch
        print("✅ Dependencies found.")
    except ImportError:
        print("🔍 Dependency missing. Attempting to install 'sentence-transformers' and 'torch'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "torch"])
            print("🚀 Dependencies installed successfully. Restarting imports...")
        except Exception as e:
            print(f"❌ Failed to install dependencies: {e}")
            print("Please run: pip install sentence-transformers torch")
            sys.exit(1)

# Check dependencies before importing heavy libraries
check_dependencies()

from sentence_transformers import SentenceTransformer
import torch

# -------- Configuration --------
CHUNKS_DIR = "chunks"
EMBEDDINGS_DIR = "embeddings"
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Ensure output directory exists
if not os.path.exists(EMBEDDINGS_DIR):
    os.makedirs(EMBEDDINGS_DIR)

def main():
    print(f"Loading embedding model: {MODEL_NAME}...")
    
    # Check if GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load the model
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # Find all JSON chunk files
    chunk_files = glob.glob(os.path.join(CHUNKS_DIR, "*.json"))
    
    if not chunk_files:
        print(f"❌ No JSON files found in the '{CHUNKS_DIR}' directory.")
        return
        
    print(f"🔎 Found {len(chunk_files)} chunk file(s) to process.")
    
    for file_path in chunk_files:
        print(f"\n{'='*60}")
        print(f"⏳ Processing {file_path}...")
        
        # Load the chunks
        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                chunks = json.load(f)
            except json.JSONDecodeError as e:
                print(f"❌ Error reading {file_path}: {e}")
                continue
                
        if not chunks:
            print(f"⚠️ No data in {file_path}, skipping.")
            continue
            
        # Prepare text for embedding
        texts_to_embed = []
        for chunk in chunks:
            text = chunk.get("text", "")
            context = chunk.get("context", "")
            
            # Combine context and text for richer semantic representation
            # If context is empty, it just uses the text
            combined_text = f"{context} > {text}" if context else text
            texts_to_embed.append(combined_text)
            
        print(f"⚡ Generating embeddings for {len(texts_to_embed)} chunks...")
        # model.encode processes the list of strings and returns a Numpy array
        embeddings = model.encode(texts_to_embed, show_progress_bar=True, convert_to_numpy=True)
        
        # Output list to store the embedded chunks
        embedded_chunks = []
        
        # Add embeddings back to the respective chunks
        for i, chunk in enumerate(chunks):
            # Create a new dictionary to avoid mutating the original in unexpected ways, 
            # though direct mutation is also fine.
            new_chunk = chunk.copy()
            # Convert numpy array back to list so it can be JSON serialized
            new_chunk["embedding"] = embeddings[i].tolist()
            embedded_chunks.append(new_chunk)
            
        output_filename = os.path.basename(file_path)
        output_path = os.path.join(EMBEDDINGS_DIR, output_filename)
        
        print(f"💾 Saving embeddings to {output_path}...")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(embedded_chunks, f, indent=4, ensure_ascii=False)
            
    print(f"\n✅ All files processed successfully! Embeddings saved in the '{EMBEDDINGS_DIR}' directory.")

if __name__ == "__main__":
    main()
