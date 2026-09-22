import os
import json
import numpy as np
import pypdf
import torch
from sentence_transformers import SentenceTransformer

# 1. Extract text from paper2.pdf
print("Parsing paper2.pdf...")
reader = pypdf.PdfReader("paper2.pdf")
raw_text = ""
for page in reader.pages:
    raw_text += page.extract_text() + "\n"

# Fallback to PyTorch OCR if normal text extraction yields nothing
if len(raw_text.strip()) < 50:
    print("pypdf extracted less than 50 characters. Falling back to RapidOCR...")
    with open("paper2.pdf", "rb") as f:
        file_bytes = f.read()
    import pypdfium2 as pdfium
    from rapidocr import RapidOCR
    from rapidocr.utils.parse_parameters import EngineType
    
    doc_pdfium = pdfium.PdfDocument(file_bytes)
    ocr_engine = RapidOCR(params={
        'Det.engine_type': EngineType.TORCH,
        'Cls.engine_type': EngineType.TORCH,
        'Rec.engine_type': EngineType.TORCH
    })
    
    ocr_lines = []
    for page in doc_pdfium:
        bitmap = page.render(scale=1.5)
        pil_img = bitmap.to_pil()
        img_np = np.array(pil_img)
        ocr_out = ocr_engine(img_np)
        if ocr_out and ocr_out.txts:
            ocr_lines.extend(ocr_out.txts)
    raw_text = "\n".join(ocr_lines)

# 2. Chunk text
print(f"Total length: {len(raw_text)}")
def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

chunks = chunk_text(raw_text)
print(f"Generated {len(chunks)} chunks.")

# 3. Load embedder and encode
print("Loading model and generating embeddings...")
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True)

# 4. Save to embeddings/paper2.json
embedded_chunks = []
for i, chunk in enumerate(chunks):
    embedded_chunks.append({
        "text": chunk,
        "context": "",
        "embedding": embeddings[i].tolist()
    })

os.makedirs("embeddings", exist_ok=True)
with open("embeddings/paper2.json", "w", encoding="utf-8") as f:
    json.dump(embedded_chunks, f, indent=4, ensure_ascii=False)
print("Saved embeddings to embeddings/paper2.json")

# 5. Build/Rebuild FAISS
print("Running build_faiss.py to update FAISS database...")
import sys
sys.path.append(os.getcwd())
import build_faiss
build_faiss.main()
print("All done!")
