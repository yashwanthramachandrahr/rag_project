import os
import sys
import json
import subprocess
import time

def check_dependencies():
    """Checks and installs required dependencies for Local LLM inference."""
    try:
        import faiss
        import numpy as np
        import sentence_transformers
        import torch
        from huggingface_hub import hf_hub_download
        from gpt4all import GPT4All
    except ImportError:
        print("🔍 Dependency missing for Generation. Attempting to install 'gpt4all' and 'huggingface_hub'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "gpt4all"])
            print("🚀 Dependencies installed successfully. Restarting imports...")
        except Exception as e:
            print(f"\n❌ Failed to install dependencies.")
            print(f"Error: {e}")
            sys.exit(1)

check_dependencies()

from huggingface_hub import hf_hub_download
from gpt4all import GPT4All
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# -------- Configuration --------
FAISS_DB_DIR = "faiss_db"
INDEX_PATH = os.path.join(FAISS_DB_DIR, "vector_store.index")
METADATA_PATH = os.path.join(FAISS_DB_DIR, "metadata.json")

EMBEDDER_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Local LLM Configuration
LLM_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
LLM_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODELS_DIR = "models"

def download_model():
    """Downloads the Mistral 7B GGUF model if not present."""
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    expected_path = os.path.join(MODELS_DIR, LLM_FILENAME)
    
    if os.path.exists(expected_path):
        print(f"✅ Found cached model: {expected_path}")
        return expected_path
        
    print(f"\n📥 Downloading Quantized TinyLlama 1.1B Model (~700 MB). This will take some time...")
    print(f"Downloading from HuggingFace -> {LLM_REPO}/{LLM_FILENAME}")
    try:
        model_path = hf_hub_download(
            repo_id=LLM_REPO,
            filename=LLM_FILENAME,
            local_dir=MODELS_DIR,
            local_dir_use_symlinks=False # Force actual download to models dir
        )
        print("✅ Download Complete!")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download model: {e}")
        sys.exit(1)

def load_database():
    """Loads FAISS index and parallel metadata."""
    if not os.path.exists(INDEX_PATH) or not os.path.exists(METADATA_PATH):
        print("❌ FAISS database not found! Please run 'python build_faiss.py' first.")
        sys.exit(1)
        
    print(f"📥 Loading FAISS index...")
    index = faiss.read_index(INDEX_PATH)
    
    print(f"📥 Loading metadata...")
    with open(METADATA_PATH, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
        
    return index, metadata

def construct_prompt(context_text, user_query):
    """
    Constructs the prompt required by TinyLlama using ChatML-like formatting.
    """
    prompt = f"""<|system|>
You are an intelligent Academic Assistant answering questions based solely on the provided Context. 
If the answer is not contained within the Context, politely state that you do not know. 
Do not hallucinate external information. Keep your answer concise and structured.</s>
<|user|>
CONTEXT:
{context_text}

QUESTION:
{user_query}</s>
<|assistant|>
"""
    return prompt

def main():
    print("="*70)
    print("  🚀 Starting End-to-End Local Python RAG System (TinyLlama-1.1B) 🚀")
    print("="*70)
    
    # 1. Download/setup local model
    llm_path = download_model()
    
    # 2. Boot up local LLM Engine
    print(f"\n🧠 Booting Local LLM (TinyLlama 1.1B) via GPT4All...")
    try:
        # GPT4All automatically uses the best available backend (CPU/Vulkan GPU)
        llm = GPT4All(
            model_name=LLM_FILENAME,
            model_path=MODELS_DIR,
            device='cpu',
            allow_download=False 
        )
        print("✅ Local LLM Initialized!")
    except Exception as e:
        print(f"❌ Failed to load LLM: {e}")
        sys.exit(1)
    
    # 3. Setup Semantic Embedder
    print(f"\n🧠 Loading Document Embedder Engine...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    
    # 4. Load Database
    index, metadata = load_database()
    print(f"✅ RAG Engine Ready! {index.ntotal} documents searchable.")
    
    # 5. Interactive Chat Loop
    while True:
        print("\n" + "="*70)
        query = input("💬 User Query (or turn off with 'quit'): ").strip()
        
        if not query:
            continue
            
        if query.lower() in ['quit', 'exit', 'q']:
            print("Shutting down Local RAG Engine. Goodbye!")
            break
            
        t0 = time.time()
        
        # --- A. Retrieval Phase ---
        print("🔍 Searching FAISS Local Database...")
        query_embedding = embedder.encode([query], convert_to_numpy=True)
        query_embedding = np.array(query_embedding, dtype=np.float32)
        
        # Pull top 3 chunks
        D, I = index.search(query_embedding, k=3)
        
        retrieved_contexts = []
        for rank in range(3):
            vector_id = int(I[0][rank])
            if vector_id == -1: continue # Handle empty results
                
            chunk_data = metadata[vector_id]
            source = chunk_data.get("source", "Unknown PDF")
            text = chunk_data.get("text", "")
            
            # Label chunks clearly for the LLM
            formatted_chunk = f"--- Document Section from '{source}' ---\n{text}\n"
            retrieved_contexts.append(formatted_chunk)
            
        combined_context = "\n".join(retrieved_contexts)
        
        if not combined_context.strip():
            print("⚠️ No relevant contextual documents found in the database. Passing straight to LLM...")
            
        # --- B. Generation Phase ---
        prompt = construct_prompt(combined_context, query)
        
        print(f"🤖 Generating Answer with TinyLlama 1.1B (Streaming)...")
        print("-" * 70)
        
        # Stream output token by token as it generates to simulate instantaneous reaction
        stream = llm.generate(
            prompt,
            max_tokens=600,
            temp=0.2, # Low temperature forces it to stick more strictly to the facts
            top_p=0.9,
            streaming=True
        )
        
        for token in stream:
            sys.stdout.write(token)
            sys.stdout.flush()
            
        t1 = time.time()
        
        print("\n" + "-" * 70)
        print(f"⏱️ Retrieval + Generation took {t1-t0:.2f} seconds.")
        print(f"📚 Sources Consulted: {len(retrieved_contexts)}")

if __name__ == "__main__":
    main()
