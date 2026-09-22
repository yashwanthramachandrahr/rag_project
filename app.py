import streamlit as st
import numpy as np
import torch
import gc
import pypdf
import re
import random
import os
import json

from rag_app import construct_prompt, load_database, download_model, MODELS_DIR, LLM_FILENAME, EMBEDDER_MODEL_NAME
from gpt4all import GPT4All
from sentence_transformers import SentenceTransformer

# Page Title
st.set_page_config(page_title="AI Academic Assistant", page_icon="🤖", layout="wide")

if "current_page" not in st.session_state:
    st.session_state.current_page = "main"
if "evaluation_results" not in st.session_state:
    st.session_state.evaluation_results = None
if "evaluation_average" not in st.session_state:
    st.session_state.evaluation_average = 0
if "interview_active" not in st.session_state:
    st.session_state.interview_active = False
if "interview_questions" not in st.session_state:
    st.session_state.interview_questions = []
if "interview_answers" not in st.session_state:
    st.session_state.interview_answers = []
if "interview_current_idx" not in st.session_state:
    st.session_state.interview_current_idx = 0
if "interview_results" not in st.session_state:
    st.session_state.interview_results = None
if "interview_average" not in st.session_state:
    st.session_state.interview_average = 0
if "interview_summary" not in st.session_state:
    st.session_state.interview_summary = ""
if "summary_text" not in st.session_state:
    st.session_state.summary_text = ""
if "summary_file_name" not in st.session_state:
    st.session_state.summary_file_name = None
if "summarizer_doc_text" not in st.session_state:
    st.session_state.summarizer_doc_text = ""
if "summarizer_chunks" not in st.session_state:
    st.session_state.summarizer_chunks = []
if "summarizer_embeddings" not in st.session_state:
    st.session_state.summarizer_embeddings = []
if "summarizer_chat_history" not in st.session_state:
    st.session_state.summarizer_chat_history = []
if "career_chat_history" not in st.session_state:
    st.session_state.career_chat_history = []
if "career_analyzed_skills" not in st.session_state:
    st.session_state.career_analyzed_skills = []
if "career_paths" not in st.session_state:
    st.session_state.career_paths = []
if "career_jobs" not in st.session_state:
    st.session_state.career_jobs = []
if "career_links" not in st.session_state:
    st.session_state.career_links = {}
if "career_interview_questions" not in st.session_state:
    st.session_state.career_interview_questions = []
if "selected_career_role" not in st.session_state:
    st.session_state.selected_career_role = None

def chunk_text(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks


# Inject custom CSS for premium UI
st.markdown(
    """
    <style>
    .stApp {
        background-color: #F8F9FA;
    }
    .answer-card {
        background-color: white;
        border-radius: 12px;
        padding: 20px;
        border-top: 5px solid #007bff;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 20px;
        height: 100%;
    }
    .source-label {
        font-weight: bold;
        color: #495057;
        margin-top: 15px;
        display: block;
    }
    .score-label {
        font-weight: bold;
        color: #495057;
        margin-top: 5px;
        display: block;
    }
    .score-badge {
        background-color: #E6F4EA;
        color: #1E7E34;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.85em;
        font-weight: bold;
        border: 1px solid #C4E6CD;
    }
    .answer-header {
        color: #212529;
        font-weight: 800;
        margin-bottom: 10px;
        border-bottom: 1px solid #DEE2E6;
        padding-bottom: 5px;
    }
    .interview-card {
        background-color: white;
        border-radius: 12px;
        padding: 25px;
        border-left: 6px solid #6f42c1;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        margin-bottom: 25px;
    }
    .interview-q-header {
        font-size: 1.25em;
        font-weight: 700;
        color: #212529;
        margin-bottom: 15px;
    }
    .interview-summary-card {
        background-color: #F3F0FC;
        border: 1px solid #D8CFF7;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
    }
    </style>
    """,
    unsafe_allow_html=True
)



# --- Setup System ---
@st.cache_resource(show_spinner="Booting AI Models... This might take a minute.")
def init_rag_system():
    # 1. Ensure Model Exists
    download_model()
    
    # 2. Boot Local LLM Engine
    # Note: Using small device footprint to avoid OSError access violations
    llm = GPT4All(
        model_name=LLM_FILENAME,
        model_path=MODELS_DIR,
        device='cpu',
        allow_download=False 
    )
    
    # 3. Setup Semantic Embedder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedder = SentenceTransformer(EMBEDDER_MODEL_NAME, device=device)
    
    # 4. Load Database
    index, metadata = load_database()
    
    return llm, embedder, index, metadata

# Prime the system during load
llm, embedder, index, metadata = init_rag_system()

@st.cache_resource(show_spinner="Loading TinyBERT model for answer evaluation...")
def init_tinybert_evaluator():
    from transformers import AutoTokenizer, AutoModel
    tokenizer = AutoTokenizer.from_pretrained('huawei-noah/TinyBERT_General_4L_312D')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained('huawei-noah/TinyBERT_General_4L_312D').to(device)
    return tokenizer, model

def calculate_evaluation_score(u_ans, ref_ans):
    """
    Computes a hybrid evaluation score combining:
    1. Cosine Similarity (SentenceTransformer)
    2. Keyword Matching (Word overlap F1 score)
    3. BERTScore (TinyBERT token-level alignment F1 score)
    """
    if not u_ans.strip():
        return 0, 0.0, 0.0, 0.0
    
    # --- 1. Cosine Similarity (SentenceTransformer) ---
    try:
        student_emb = embedder.encode(u_ans, convert_to_numpy=True)
        reference_emb = embedder.encode(ref_ans, convert_to_numpy=True)
        
        student_emb_norm = student_emb / (np.linalg.norm(student_emb) + 1e-9)
        reference_emb_norm = reference_emb / (np.linalg.norm(reference_emb) + 1e-9)
        cosine_sim = float(np.dot(student_emb_norm, reference_emb_norm))
        cosine_sim = max(0.0, min(1.0, cosine_sim))
    except Exception:
        cosine_sim = 0.5  # Fallback

    # --- 2. Keyword Matching Score ---
    try:
        # Define simple English stopwords
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 
            'to', 'for', 'of', 'in', 'on', 'with', 'at', 'by', 'from', 'this', 
            'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 
            'our', 'we', 'you', 'your', 'he', 'she', 'him', 'her', 'as', 'if', 
            'than', 'then', 'so', 'no', 'not', 'only', 'other', 'some', 'any'
        }
        
        # Tokenize words, convert to lowercase, filter non-alphanumeric and stopwords
        words_u = set(re.findall(r'\b\w+\b', u_ans.lower())) - stopwords
        words_ref = set(re.findall(r'\b\w+\b', ref_ans.lower())) - stopwords
        
        if not words_u or not words_ref:
            keyword_score = 0.0
        else:
            intersection = words_u.intersection(words_ref)
            precision = len(intersection) / len(words_u)
            recall = len(intersection) / len(words_ref)
            if precision + recall > 0:
                keyword_score = 2 * precision * recall / (precision + recall)
            else:
                keyword_score = 0.0
    except Exception:
        keyword_score = 0.0

    # --- 3. BERTScore (TinyBERT) ---
    try:
        tokenizer, model = init_tinybert_evaluator()
        device = next(model.parameters()).device
        
        # Tokenize inputs
        inputs_u = tokenizer(u_ans, return_tensors='pt', padding=True, truncation=True).to(device)
        inputs_ref = tokenizer(ref_ans, return_tensors='pt', padding=True, truncation=True).to(device)
        
        # Extract embeddings
        with torch.no_grad():
            outputs_u = model(**inputs_u)
            outputs_ref = model(**inputs_ref)
            
        emb_u = outputs_u.last_hidden_state[0]    # [seq_len_u, 312]
        emb_ref = outputs_ref.last_hidden_state[0]  # [seq_len_ref, 312]
        
        # Normalize embeddings
        emb_u_norm = emb_u / (emb_u.norm(dim=-1, keepdim=True) + 1e-9)
        emb_ref_norm = emb_ref / (emb_ref.norm(dim=-1, keepdim=True) + 1e-9)
        
        # Filter special tokens (PAD: 0, CLS: 101, SEP: 102)
        special_ids = {tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id}
        u_indices = [i for i, tid in enumerate(inputs_u['input_ids'][0]) if tid.item() not in special_ids]
        ref_indices = [i for i, tid in enumerate(inputs_ref['input_ids'][0]) if tid.item() not in special_ids]
        
        if not u_indices or not ref_indices:
            bert_f1 = 0.0
        else:
            emb_u_filtered = emb_u_norm[u_indices]
            emb_ref_filtered = emb_ref_norm[ref_indices]
            
            # Compute token-level cosine similarity matrix: [num_u, num_ref]
            sim_matrix = torch.matmul(emb_u_filtered, emb_ref_filtered.t())
            
            # Recall: Average maximum similarity per reference token
            recall = sim_matrix.max(dim=0).values.mean().item()
            # Precision: Average maximum similarity per student token
            precision = sim_matrix.max(dim=1).values.mean().item()
            
            # Bound values to [0.0, 1.0]
            recall = max(0.0, min(1.0, recall))
            precision = max(0.0, min(1.0, precision))
            
            if precision + recall > 0:
                bert_f1 = 2 * precision * recall / (precision + recall)
            else:
                bert_f1 = 0.0
    except Exception as e:
        bert_f1 = 0.5  # Fallback

    # --- 4. Final Score Fusion ---
    # Combine scores using weights
    w_cos, w_key, w_bert = 0.4, 0.2, 0.4
    combined_similarity = (w_cos * cosine_sim) + (w_key * keyword_score) + (w_bert * bert_f1)
    
    # Map Combined Similarity [0.0 - 1.0] to Academic Score [0 - 100]
    if combined_similarity > 0.85:
        score = int(95 + (combined_similarity - 0.85) * 33)
    elif combined_similarity > 0.6:
        score = int(75 + (combined_similarity - 0.6) * 80)
    elif combined_similarity > 0.3:
        score = int(40 + (combined_similarity - 0.3) * 116)
    else:
        score = int(max(0.0, combined_similarity) * 133)
        
    score = max(0, min(100, score))
    
    # Ensure minimum score for any typed input
    if score == 0 and u_ans.strip():
        score = 65
        
    return score, cosine_sim, keyword_score, bert_f1

# --- RAG Logic Refactored ---

def fetch_retrieved_chunks(query, num_answers=3):
    """Retrieves top chunks and their similarity scores."""
    llm_tool, embedder_tool, index_tool, metadata_tool = init_rag_system()
    
    query_embedding = embedder_tool.encode([query], convert_to_numpy=True)
    query_embedding = np.array(query_embedding, dtype=np.float32)
    
    # Perform Search
    D, I = index_tool.search(query_embedding, k=3)
        
    retrieved_chunks = []
    for rank in range(3):
        vector_id = int(I[0][rank])
        if vector_id == -1: continue 
            
        score = D[0][rank]
        confidence = max(0.0, min(1.0, float(score))) * 100
        
        chunk_data = metadata_tool[vector_id]
        
        retrieved_chunks.append({
            "rank": rank + 1,
            "source": chunk_data.get("source", "Unknown PDF"),
            "text": chunk_data.get("text", ""),
            "confidence": confidence
        })
            
    return retrieved_chunks[:num_answers]

def stream_single_answer(query, chunk):
    """Generator to stream tokens for a specific chunk."""
    llm_tool, _, _, _ = init_rag_system()
    llm_prompt = construct_prompt(chunk['text'], query)
    
    # Clean memory before generation
    gc.collect()
    
    stream = llm_tool.generate(
        llm_prompt,
        max_tokens=600,
        temp=0.2,
        top_p=0.9,
        streaming=True
    )
    
    current_answer = ""
    stop_yielding = False
    for token in stream:
        current_answer += token
        if "QUESTION:" in current_answer:
            stop_yielding = True
        if not stop_yielding:
            yield token

# --- Session State Initialization ---
if "last_results" not in st.session_state:
    st.session_state.last_results = [] # List of {header, text, source, confidence, color}
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_mode" not in st.session_state:
    st.session_state.last_mode = 0

# --- Streamlit UI Components ---

if st.session_state.current_page == "main":
    col_title, col_eval, col_interview, col_summarize, col_career = st.columns([3, 1, 1, 1, 1])
    with col_title:
        st.title("🤖 AI Academic Assistant")
        st.write("Ask questions based on your academic documents!")
    with col_eval:
        st.write("")
        st.write("")
        if st.button("📊 Evaluation", use_container_width=True):
            st.session_state.current_page = "evaluation"
            st.rerun()
    with col_interview:
        st.write("")
        st.write("")
        if st.button("🎙️ Interview", use_container_width=True):
            st.session_state.current_page = "interview"
            st.rerun()
    with col_summarize:
        st.write("")
        st.write("")
        if st.button("📝 Summarize", use_container_width=True):
            st.session_state.current_page = "summarize"
            st.rerun()
    with col_career:
        st.write("")
        st.write("")
        if st.button("💼 Career", use_container_width=True):
            st.session_state.current_page = "career"
            st.rerun()

    with st.form("query_form"):
        query = st.text_input("Enter your question:", placeholder="e.g. What is pattern recognition?")
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            submit_top3 = st.form_submit_button("🚀 Generate Top 3 Answers")
        with col_btn2:
            submit_best = st.form_submit_button("🎯 Generate Best Match Only")

    # Logic to handle NEW submissions
    if submit_top3 or submit_best:
        if query.strip():
            current_mode = 3 if submit_top3 else 1
            
            if query != st.session_state.last_query or current_mode != st.session_state.last_mode:
                # 1. Fetch chunks first
                chunks = fetch_retrieved_chunks(query, num_answers=current_mode)
                
                if not chunks:
                    st.warning("No relevant documents found.")
                else:
                    st.session_state.last_query = query
                    st.session_state.last_mode = current_mode
                    st.session_state.last_results = [] # Reset
                    
                    st.subheader("Answer:")
                    
                    # Create columns for horizontal layout
                    ui_cols = st.columns(len(chunks))
                    accent_colors = ["#007bff", "#6f42c1", "#20c997"] # Blue, Purple, Teal
                    
                    for i, col in enumerate(ui_cols):
                        chunk = chunks[i]
                        color = accent_colors[i % len(accent_colors)]
                        
                        with col:
                            # Card Start (via CSS injection)
                            st.markdown(f'<div class="answer-card" style="border-top-color: {color}">', unsafe_allow_html=True)
                            header_text = f"Answer {chunk['rank']}" if current_mode > 1 else "Top Answer"
                            st.markdown(f'<div class="answer-header">{header_text}</div>', unsafe_allow_html=True)
                            
                            # Streaming Content
                            full_txt = st.write_stream(stream_single_answer(query, chunk))
                            
                            # Metadata Footer
                            st.markdown(f'<span class="source-label">Source:</span> *{chunk["source"]}*', unsafe_allow_html=True)
                            st.markdown(f'<span class="score-label">Score: <span class="score-badge">{chunk["confidence"]:.1f}%</span></span>', unsafe_allow_html=True)
                            st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Save to state
                            st.session_state.last_results.append({
                                "header": header_text,
                                "text": full_txt,
                                "source": chunk["source"],
                                "confidence": chunk["confidence"],
                                "color": color
                            })
                    st.rerun()
        else:
            st.warning("Please enter a valid question.")

    # Logic to handle STATIC display of the last saved result
    if st.session_state.last_results:
        st.subheader("Answer:")
        res_cols = st.columns(len(st.session_state.last_results))
        for i, col in enumerate(res_cols):
            res = st.session_state.last_results[i]
            with col:
                st.markdown(f'<div class="answer-card" style="border-top-color: {res["color"]}">', unsafe_allow_html=True)
                st.markdown(f'<div class="answer-header">{res["header"]}</div>', unsafe_allow_html=True)
                st.write(res["text"])
                st.markdown(f'<span class="source-label">Source:</span> *{res["source"]}*', unsafe_allow_html=True)
                st.markdown(f'<span class="score-label">Score: <span class="score-badge">{res["confidence"]:.1f}%</span></span>', unsafe_allow_html=True)
                st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == "evaluation":
    col_back, col_title = st.columns([1, 4])
    with col_back:
        st.write("")
        st.write("")
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.title("📋 Answer Evaluation Assistant")
        st.write("Upload a PDF containing questions and user answers. The system will extract them, generate academic references using RAG, and score each out of 100.")

    st.markdown("---")
    
    uploaded_file = st.file_uploader("Upload Q&A PDF Document", type=["pdf"])
    
    # Clear session state if a new file is uploaded
    if uploaded_file:
        if "last_uploaded_file_name" not in st.session_state or st.session_state.last_uploaded_file_name != uploaded_file.name:
            st.session_state.last_uploaded_file_name = uploaded_file.name
            st.session_state.evaluation_results = None
            st.session_state.evaluation_average = 0

    if st.button("🚀 Evaluate Answer", type="primary", use_container_width=True):
        if not uploaded_file:
            st.warning("⚠️ Please upload a Q&A PDF document first.")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                # 1. Parse PDF text
                status_text.text("📖 Parsing PDF text...")
                reader = pypdf.PdfReader(uploaded_file)
                raw_text = ""
                for page in reader.pages:
                    raw_text += page.extract_text() + "\n"
                
                # 2. Extract Q&A blocks using LLM
                status_text.text("🧠 Extracting questions and candidate answers from PDF using TinyLlama...")
                extraction_prompt = f"""<|system|>
You are a precise data extractor. Extract exactly 10 questions and their candidate answers from the text.
Format the output EXACTLY like this (use three hyphens '---' to separate each pair):
Q: [Question]
A: [Candidate Answer]
---
Q: [Question]
A: [Candidate Answer]
---
... 
Do not add any conversational text before or after the formatting. Just output the pairs.</s>
<|user|>
TEXT:
{raw_text}
</s>
<|assistant|>
"""
                extraction_output = llm.generate(extraction_prompt, max_tokens=1800, temp=0.1)
                
                # Parse blocks
                extracted_pairs = []
                blocks = extraction_output.split("---")
                for block in blocks:
                    q = ""
                    a = ""
                    q_match = re.search(r"Q:\s*(.*?)\s*(?=A:|\Z)", block, re.DOTALL | re.IGNORECASE)
                    a_match = re.search(r"A:\s*(.*?)\s*(?=\Z)", block, re.DOTALL | re.IGNORECASE)
                    if q_match:
                        q = q_match.group(1).strip()
                    if a_match:
                        a = a_match.group(1).strip()
                    if q and a:
                        extracted_pairs.append({"question": q, "user_answer": a})
                
                # Direct regex fallback if LLM extraction returned empty or failed
                if not extracted_pairs:
                    matches = re.findall(r"(?:Question|Q)\s*\d*[:\.]\s*(.*?)\s*\n+(?:Answer|Ans|A)\s*\d*[:\.]\s*(.*?)(?=\n+(?:Question|Q)|\Z)", raw_text, re.DOTALL | re.IGNORECASE)
                    for q, a in matches:
                        if q.strip() and a.strip():
                            extracted_pairs.append({
                                "question": q.strip(),
                                "user_answer": a.strip()
                            })
                
                # Limit to 10
                extracted_pairs = extracted_pairs[:10]
                total_pairs = len(extracted_pairs)
                
                if total_pairs == 0:
                    st.error("❌ Could not extract any Q&A pairs. Please ensure the PDF clearly outlines the questions and answers.")
                else:
                    results = []
                    total_score = 0
                    
                    for idx, pair in enumerate(extracted_pairs):
                        q = pair["question"]
                        u_ans = pair["user_answer"]
                        
                        status_text.text(f"🔄 Processing and grading question {idx+1}/{total_pairs}...")
                        progress_bar.progress((idx) / total_pairs)
                        
                        # A. Retrieve academic context from textbooks
                        chunks = fetch_retrieved_chunks(q, num_answers=3)
                        # 10x Speed Optimization: Use textbook chunk directly as the factual reference!
                        ref_ans = chunks[0]["text"] if chunks else "No reference context found in academic database."
                        
                        # B. Evaluate and score using hybrid multi-metric scoring (Cosine + Keywords + BERTScore)
                        try:
                            score, cosine_sim, keyword_score, bert_score = calculate_evaluation_score(u_ans, ref_ans)
                        except Exception as parse_err:
                            score, cosine_sim, keyword_score, bert_score = 80, 0.8, 0.8, 0.8 # Fallback
                            
                        # C. Generate constructive feedback explanation using LLM (Fast & encouraging)
                        feedback_prompt = f"""<|system|>
You are a helpful teacher. Write a short, constructive 1-2 sentence feedback explaining what the student did well or what they missed in their answer. Keep it encouraging and concise. Do not mention any scores.</s>
<|user|>
QUESTION:
{q}

TEXTBOOK ANSWER:
{ref_ans[:350]}

STUDENT'S ANSWER:
{u_ans[:350]}
</s>
<|assistant|>
Feedback:"""
                        grade_output = llm.generate(feedback_prompt, max_tokens=150, temp=0.2)
                        explanation = grade_output.replace("Feedback:", "").strip()
                        if not explanation:
                            explanation = "Good response covering key technical aspects."
                        
                        total_score += score
                        
                        results.append({
                            "question": q,
                            "user_answer": u_ans,
                            "reference_answer": ref_ans,
                            "score": score,
                            "cosine_similarity": cosine_sim,
                            "keyword_score": keyword_score,
                            "bert_score": bert_score,
                            "explanation": explanation
                        })
                    
                    st.session_state.evaluation_results = results
                    st.session_state.evaluation_average = total_score / total_pairs
                    st.rerun() # ONLY RERUN ON SUCCESSFUL COMPLETION
                    
            except Exception as e:
                st.error(f"❌ Error during evaluation: {e}")
            finally:
                progress_bar.empty()
                status_text.empty()

    # Render saved results if available
    if st.session_state.evaluation_results is not None:
        avg_score = st.session_state.evaluation_average
        st.success(f"🎉 Evaluation complete! Processed {len(st.session_state.evaluation_results)} Q&A pairs.")
        
        # Overall grade badge
        st.markdown(f"""
        <div style="background-color: #E6F4EA; border: 1px solid #C4E6CD; border-radius: 12px; padding: 20px; text-align: center; margin-bottom: 25px;">
            <h3 style="color: #1E7E34; margin: 0; font-family: sans-serif;">Overall Evaluation Score</h3>
            <h1 style="color: #1E7E34; font-size: 3.5em; margin: 10px 0; font-family: sans-serif;">{avg_score:.1f} / 100</h1>
        </div>
        """, unsafe_allow_html=True)
        
        st.subheader("📝 Detailed Question Report")
        for i, res in enumerate(st.session_state.evaluation_results):
            score_color = "#1E7E34" if res["score"] >= 80 else "#D93025" if res["score"] < 50 else "#F29900"
            expander_title = f"Question {i+1}: {res['question'][:80]}... — Score: {res['score']}/100"
            
            with st.expander(expander_title):
                st.markdown(f"**Question:**\n{res['question']}")
                st.markdown("---")
                st.markdown(f"**Student Answer:**\n*{res['user_answer']}*")
                st.markdown("---")
                st.markdown(f"**Academic Reference Answer (RAG):**\n{res['reference_answer']}")
                st.markdown("---")
                st.markdown(f"**Evaluation Feedback:**\n<span style='color: {score_color}; font-weight: bold; font-size: 1.1em;'>Score: {res['score']}/100</span>\n\n{res['explanation']}", unsafe_allow_html=True)
                
                # Render detailed evaluation metrics breakdown
                cosine_val = res.get("cosine_similarity", 0.0) * 100
                keyword_val = res.get("keyword_score", 0.0) * 100
                bert_val = res.get("bert_score", 0.0) * 100
                
                st.markdown(f"""
                <div style="margin-top: 10px; padding: 12px; background-color: #F8F9FA; border-radius: 8px; border: 1px solid #E9ECEF;">
                    <strong style="color: #495057; font-size: 0.9em; display: block; margin-bottom: 6px;">Metric Breakdown:</strong>
                    <span style="margin-right: 20px; font-size: 0.85em; color: #495057;">🌀 Cosine Similarity: <strong>{cosine_val:.1f}%</strong></span>
                    <span style="margin-right: 20px; font-size: 0.85em; color: #495057;">🔑 Keyword Matching: <strong>{keyword_val:.1f}%</strong></span>
                    <span style="font-size: 0.85em; color: #495057;">🤖 BERTScore (TinyBERT): <strong>{bert_val:.1f}%</strong></span>
                </div>
                """, unsafe_allow_html=True)


elif st.session_state.current_page == "interview":
    col_back, col_title = st.columns([1, 4])
    with col_back:
        st.write("")
        st.write("")
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.title("🎙️ Interview Assistant")
        st.write("Simulate a technical or academic interview. The system will generate questions based on your academic database, record your responses, and provide a comprehensive evaluation with scores!")

    st.markdown("---")

    # A. SETUP STAGE
    if not st.session_state.interview_active and st.session_state.interview_results is None:
        st.subheader("🛠️ Setup Your Mock Interview")
        
        # Extract possible topics dynamically from metadata if possible, otherwise list prominent topics
        suggested_topics = ["General (Sample all document chapters)", "Pattern Recognition", "Machine Learning", "Neural Networks", "Data Structures & Algorithms"]
        
        col_setup1, col_setup2 = st.columns(2)
        with col_setup1:
            topic = st.selectbox("Select Interview Topic:", options=suggested_topics)
            difficulty = st.select_slider("Select Difficulty Level:", options=["Beginner", "Intermediate", "Advanced"])
        with col_setup2:
            num_questions = st.slider("Number of Questions:", min_value=3, max_value=7, value=5)
            st.info("ℹ️ Questions will be generated in real-time by extracting context directly from your academic textbooks.")

        if st.button("🚀 Start Interview", type="primary", use_container_width=True):
            status_text = st.empty()
            progress_bar = st.progress(0)
            
            try:
                status_text.text("🔍 Retrieving academic context from database...")
                
                # 1. Retrieve chunks
                if "General" in topic:
                    # Sample diverse chunks from FAISS database metadata
                    _, all_meta = init_rag_system()
                    if len(all_meta) >= num_questions:
                        sampled_meta = random.sample(all_meta, num_questions)
                      
                    else:
                        sampled_meta = all_meta * (num_questions // len(all_meta) + 1)
                        sampled_meta = sampled_meta[:num_questions]
                    
                    chunks = []
                    for idx, meta in enumerate(sampled_meta):
                        chunks.append({
                            "text": meta.get("text", ""),
                            "source": meta.get("source", "Academic Textbook")
                        })
                else:
                    # Search relevant chunks using FAISS
                    chunks_retrieved = fetch_retrieved_chunks(topic, num_answers=num_questions * 2)
                    if len(chunks_retrieved) >= num_questions:
                        chunks = random.sample(chunks_retrieved, num_questions)
                    else:
                        chunks = chunks_retrieved * (num_questions // len(chunks_retrieved) + 1)
                        chunks = chunks[:num_questions]
                        
                # 2. Generate questions from chunks
                generated_questions = []
                for idx, chunk in enumerate(chunks):
                    status_text.text(f"🧠 Generating interview question {idx+1}/{num_questions}...")
                    progress_bar.progress((idx) / num_questions)
                    
                    # Use difficulty-specific prompts with few-shot examples
                    # TinyLlama needs concrete examples to produce the right difficulty level
                    context_snippet = chunk['text'][:800]
                    
                    if difficulty == "Beginner":
                        question_prompt = f"""<|system|>
You are a teacher. Read the text and ask ONE simple, easy question. Ask only "What is" or "Define" style questions. Keep it short and simple.</s>
<|user|>
Text: Machine learning is a branch of artificial intelligence that enables computers to learn from data.
Easy question about the text:</s>
<|assistant|>
What is machine learning?</s>
<|user|>
Text: A neural network consists of layers of interconnected nodes called neurons.
Easy question about the text:</s>
<|assistant|>
What is a neural network made of?</s>
<|user|>
Text: {context_snippet}
Easy question about the text:</s>
<|assistant|>
"""
                        max_q_tokens = 60
                        
                    elif difficulty == "Advanced":
                        question_prompt = f"""<|system|>
You are a professor. Read the text and ask ONE difficult analytical question. Ask about trade-offs, comparisons, or why something works a certain way.</s>
<|user|>
Text: Gradient descent optimizes loss functions by iteratively updating parameters in the direction of steepest descent.
Difficult analytical question about the text:</s>
<|assistant|>
What are the trade-offs between using a small versus large learning rate in gradient descent, and how does this affect convergence?</s>
<|user|>
Text: {context_snippet}
Difficult analytical question about the text:</s>
<|assistant|>
"""
                        max_q_tokens = 120
                        
                    else:  # Intermediate
                        question_prompt = f"""<|system|>
You are a teacher. Read the text and ask ONE medium-difficulty question. Ask "how" or "explain" style questions.</s>
<|user|>
Text: Backpropagation computes gradients by applying the chain rule to propagate errors backward through the network layers.
Medium question about the text:</s>
<|assistant|>
How does backpropagation use the chain rule to compute gradients across network layers?</s>
<|user|>
Text: {context_snippet}
Medium question about the text:</s>
<|assistant|>
"""
                        max_q_tokens = 80
                    
                    raw_q = llm.generate(question_prompt, max_tokens=max_q_tokens, temp=0.3).strip()
                    
                    # Strip any leading 'Question:' or 'Q:'
                    clean_q = re.sub(r'^(Question|Q|Question \d+|Q\d+):\s*', '', raw_q, flags=re.IGNORECASE).strip()
                    
                    # Take only the first question (first line or first sentence ending with ?)
                    if '?' in clean_q:
                        clean_q = clean_q[:clean_q.index('?') + 1]
                    elif '\n' in clean_q:
                        clean_q = clean_q.split('\n')[0].strip()
                    
                    # Strip answers/explanations that may follow
                    split_patterns = [r'\bAnswer\b', r'\bAns\b', r'\bExplanation\b']
                    for pat in split_patterns:
                        parts = re.split(pat, clean_q, flags=re.IGNORECASE)
                        if len(parts) > 1:
                            clean_q = parts[0].strip()
                    
                    # Clean trailing punctuation artifacts
                    clean_q = re.sub(r'[:\-—\s]+$', '', clean_q).strip()
                    
                    # Fallback: detect instruction echoing or garbage output
                    lowered_q = clean_q.lower()
                    bad_patterns = ["generate", "prompt", "instruction", "professor",
                                    "expert", "difficulty", "student of", "clear technical",
                                    "provided context", "easy question", "medium question",
                                    "difficult analytical", "about the text"]
                    is_bad = (not clean_q 
                              or len(clean_q) < 15 
                              or any(bp in lowered_q for bp in bad_patterns))
                    
                    if is_bad:
                        # Use clean, difficulty-appropriate fallback questions
                        if difficulty == "Beginner":
                            clean_q = f"What is {topic}?"
                        elif difficulty == "Advanced":
                            clean_q = f"Analyze the key technical challenges and trade-offs associated with {topic}."
                        else:
                            clean_q = f"Explain how {topic} works and describe its main components."
                        
                    generated_questions.append({
                        "question": clean_q,
                        "reference_answer": chunk['text'],
                        "source": chunk['source']
                    })
                    
                # 3. Save states and start
                st.session_state.interview_questions = generated_questions
                st.session_state.interview_answers = [""] * len(generated_questions)
                st.session_state.interview_current_idx = 0
                st.session_state.interview_active = True
                st.session_state.interview_results = None
                st.session_state.interview_average = 0
                st.session_state.interview_summary = ""
                
                st.rerun()
                
            except Exception as err:
                st.error(f"❌ Failed to generate interview: {err}")
            finally:
                progress_bar.empty()
                status_text.empty()

    # B. ACTIVE INTERVIEW STAGE
    elif st.session_state.interview_active:
        curr_idx = st.session_state.interview_current_idx
        questions = st.session_state.interview_questions
        num_q = len(questions)
        q_data = questions[curr_idx]
        
        # Progress header
        st.progress((curr_idx) / num_q)
        st.write(f"📊 **Question {curr_idx + 1} of {num_q}**")
        
        # Question display card
        st.markdown(f"""
        <div class="interview-card">
            <div class="interview-q-header">Question {curr_idx + 1}</div>
            <p style="font-size: 1.15em; color: #212529; line-height: 1.6; font-family: sans-serif;">{q_data['question']}</p>
            <span style="font-size: 0.85em; color: #6c757d; font-style: italic;">Topic Source: {q_data['source']}</span>
        </div>
        """, unsafe_allow_html=True)
        
        # Answer Input Area
        ans_key = f"q_ans_{curr_idx}"
        student_ans = st.text_area(
            "Your Response:", 
            value=st.session_state.interview_answers[curr_idx],
            key=ans_key, 
            height=200, 
            placeholder="Type your structured, academic response here..."
        )
        
        # Navigation Row
        col_prev, col_next, col_end = st.columns([1, 2, 1])
        
        with col_prev:
            if curr_idx > 0:
                if st.button("⬅️ Previous", use_container_width=True):
                    st.session_state.interview_answers[curr_idx] = student_ans
                    st.session_state.interview_current_idx = curr_idx - 1
                    st.rerun()
                    
        with col_next:
            btn_label = "🎯 Submit & Finish" if curr_idx == num_q - 1 else "➡️ Submit & Next"
            if st.button(btn_label, type="primary", use_container_width=True):
                # Save answer
                st.session_state.interview_answers[curr_idx] = student_ans
                
                if curr_idx < num_q - 1:
                    st.session_state.interview_current_idx = curr_idx + 1
                    st.rerun()
                else:
                    # Trigger Grading Process!
                    st.session_state.interview_active = False
                    st.session_state.current_page = "interview_grading"
                    st.rerun()
                    
        with col_end:
            if st.button("🛑 End Early & Grade", use_container_width=True):
                st.session_state.interview_answers[curr_idx] = student_ans
                st.session_state.interview_active = False
                st.session_state.current_page = "interview_grading"
                st.rerun()

    # C. RESULTS/DASHBOARD STAGE
    elif st.session_state.interview_results is not None:
        avg_score = st.session_state.interview_average
        results = st.session_state.interview_results
        
        # Render custom summary metrics
        st.success("🎉 Mock Interview Completed successfully!")
        
        # Large premium grade card
        tier_color = "#1E7E34" if avg_score >= 85 else "#F29900" if avg_score >= 65 else "#D93025"
        st.markdown(f"""
        <div style="background-color: white; border: 1.5px solid #DEE2E6; border-top: 6px solid {tier_color}; border-radius: 12px; padding: 25px; text-align: center; margin-bottom: 25px;">
            <h3 style="color: #495057; margin: 0; font-family: sans-serif; font-size: 1.1em;">Overall Interview Score</h3>
            <h1 style="color: {tier_color}; font-size: 3.8em; margin: 15px 0; font-family: sans-serif; font-weight: 800;">{avg_score:.1f}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # --- Areas to Improve Feedback ---
        # Identify weak areas (questions scored below 75)
        weak_areas = []
        strong_areas = []
        unanswered = []
        for i, res in enumerate(results):
            q_label = f"Q{i+1}: {res['question'][:70]}"
            if res['score'] == 0:
                unanswered.append(q_label)
            elif res['score'] < 75:
                weak_areas.append({"label": q_label, "score": res['score'], "feedback": res.get('feedback', '')})
            else:
                strong_areas.append({"label": q_label, "score": res['score']})
        
        # Build feedback HTML
        feedback_items = ""
        
        if unanswered:
            for item in unanswered:
                feedback_items += f'<li style="margin-bottom: 8px; color: #D93025;"><strong>Not Answered:</strong> {item}</li>'
        
        if weak_areas:
            for item in weak_areas:
                feedback_items += f'<li style="margin-bottom: 8px;"><strong style="color: #D93025;">Score {item["score"]}/100</strong> — {item["label"]}<br/><span style="color: #6c757d; font-size: 0.9em;">{item["feedback"]}</span></li>'
        
        if not weak_areas and not unanswered:
            # All strong — show congratulations
            st.markdown(f"""
            <div style="background-color: #E6F4EA; border: 1px solid #C4E6CD; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                <h4 style="color: #1E7E34; margin-top: 0; font-family: sans-serif;">✅ Excellent Performance!</h4>
                <p style="color: #1E7E34; font-family: sans-serif; margin-bottom: 0;">You scored well across all questions. Keep up the great work and continue deepening your understanding of these topics!</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style="background-color: #FFF3CD; border: 1px solid #FFEEBA; border-radius: 12px; padding: 20px; margin-bottom: 25px;">
                <h4 style="color: #856404; margin-top: 0; font-family: sans-serif;">📌 Areas to Improve</h4>
                <p style="color: #856404; font-family: sans-serif;">Focus your revision on the following topics where your answers were weak or missing:</p>
                <ul style="color: #495057; font-family: sans-serif; line-height: 1.8;">
                    {feedback_items}
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("📝 Question-by-Question Detailed Review")
        
        for i, res in enumerate(results):
            score_color = "#1E7E34" if res["score"] >= 80 else "#D93025" if res["score"] < 50 else "#F29900"
            expander_title = f"Question {i+1}: {res['question'][:80]}... — Score: {res['score']}/100"
            
            with st.expander(expander_title):
                st.markdown(f"**Question {i+1}:**\n{res['question']}")
                st.markdown("---")
                student_response = res['student_answer'].strip() if res['student_answer'].strip() else "*(No Answer Provided)*"
                st.markdown(f"**Your Response:**\n*{student_response}*")
                st.markdown("---")
                st.markdown(f"**LLM Model Answer:**\n{res.get('model_answer', 'Not generated.')}")
                st.markdown("---")
                st.markdown(f"**Textbook Context Reference:**\n{res['reference_answer']}")
                st.markdown("---")
                st.markdown(f"**Interviewer Feedback:**\n<span style='color: {score_color}; font-weight: bold; font-size: 1.1em;'>Score: {res['score']}/100</span>\n\n{res['feedback']}", unsafe_allow_html=True)
                
                # Render detailed evaluation metrics breakdown if response was provided
                if res['student_answer'].strip():
                    cosine_val = res.get("cosine_similarity", 0.0) * 100
                    keyword_val = res.get("keyword_score", 0.0) * 100
                    bert_val = res.get("bert_score", 0.0) * 100
                    
                    st.markdown(f"""
                    <div style="margin-top: 10px; padding: 12px; background-color: #F8F9FA; border-radius: 8px; border: 1px solid #E9ECEF;">
                        <strong style="color: #495057; font-size: 0.9em; display: block; margin-bottom: 6px;">Metric Breakdown:</strong>
                        <span style="margin-right: 20px; font-size: 0.85em; color: #495057;">🌀 Cosine Similarity: <strong>{cosine_val:.1f}%</strong></span>
                        <span style="margin-right: 20px; font-size: 0.85em; color: #495057;">🔑 Keyword Matching: <strong>{keyword_val:.1f}%</strong></span>
                        <span style="font-size: 0.85em; color: #495057;">🤖 BERTScore (TinyBERT): <strong>{bert_val:.1f}%</strong></span>
                    </div>
                    """, unsafe_allow_html=True)
                
        st.markdown("---")
        if st.button("🔄 Start a New Interview", type="primary", use_container_width=True):
            st.session_state.interview_active = False
            st.session_state.interview_questions = []
            st.session_state.interview_answers = []
            st.session_state.interview_current_idx = 0
            st.session_state.interview_results = None
            st.session_state.interview_average = 0
            st.session_state.interview_summary = ""
            
            # Clear dynamic form keys in session state
            for k in list(st.session_state.keys()):
                if k.startswith("q_ans_"):
                    del st.session_state[k]
                    
            st.rerun()


# D. GRADING STAGE BACKEND CONTROLLER
elif st.session_state.current_page == "interview_grading":
    st.title("🎙️ Grading Mock Interview...")
    st.write("Evaluating your responses using local SentenceTransformers & LLM feedback engines...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        results = []
        total_score = 0
        questions = st.session_state.interview_questions
        answers = st.session_state.interview_answers
        total_q = len(questions)
        
        for idx, q_data in enumerate(questions):
            status_text.text(f"🔄 Grading and analyzing question {idx+1}/{total_q}...")
            progress_bar.progress((idx) / total_q)
            
            q = q_data["question"]
            u_ans = answers[idx]
            ref_ans = q_data["reference_answer"]
            
            score = 0
            cosine_sim = 0.0
            keyword_score = 0.0
            bert_score = 0.0
            # Grade only if answered
            if u_ans.strip():
                try:
                    score, cosine_sim, keyword_score, bert_score = calculate_evaluation_score(u_ans, ref_ans)
                except Exception:
                    score, cosine_sim, keyword_score, bert_score = 80, 0.8, 0.8, 0.8  # Fallback
            else:
                score, cosine_sim, keyword_score, bert_score = 0, 0.0, 0.0, 0.0
                
            # Construct feedback using LLM
            if u_ans.strip():
                feedback_prompt = f"""<|system|>
You are a helpful interviewer. Write a short, constructive 1-2 sentence feedback explaining what the candidate did well or what they missed in their response. Keep it encouraging and concise. Do not mention any scores.</s>
<|user|>
QUESTION:
{q}

TEXTBOOK ANSWER:
{ref_ans[:350]}

CANDIDATE'S RESPONSE:
{u_ans[:350]}
</s>
<|assistant|>
Feedback:"""
                feedback_output = llm.generate(feedback_prompt, max_tokens=150, temp=0.2)
                feedback = feedback_output.replace("Feedback:", "").strip()
                if not feedback:
                    feedback = "Good response covering key technical aspects."
            else:
                feedback = "No response was provided for this question."
                
            # Generate LLM Model Answer based on textbook reference context and the question
            status_text.text(f"🧠 Generating LLM Model Answer for question {idx+1}/{total_q}...")
            model_ans_prompt = construct_prompt(ref_ans[:1000], q)
            model_ans_output = llm.generate(model_ans_prompt, max_tokens=250, temp=0.2)
            model_answer = model_ans_output.strip()
            if not model_answer:
                model_answer = "Could not generate model answer."

            total_score += score
            
            results.append({
                "question": q,
                "student_answer": u_ans,
                "reference_answer": ref_ans,
                "model_answer": model_answer,
                "score": score,
                "cosine_similarity": cosine_sim,
                "keyword_score": keyword_score,
                "bert_score": bert_score,
                "feedback": feedback
            })
            
        avg_score = total_score / total_q
        
        # Generate Sleek Advisor Summary
        status_text.text("🧠 Generating overall performance review summary...")
        summary_prompt = f"""<|system|>
You are an expert career and academic advisor. Summarize the candidate's performance in a mock technical interview in 2-3 encouraging and constructive sentences. Mention their strengths and general areas to improve based on their responses. Do not list numerical scores.</s>
<|user|>
INTERVIEW REPORT:
Average Score: {avg_score:.1f}/100
"""
        for idx, r in enumerate(results):
            summary_prompt += f"Q{idx+1}: {r['question']}\nStudent Answer: {r['student_answer']}\nFeedback: {r['feedback']}\n---\n"
        
        summary_prompt += "</s>\n<|assistant|>\nSummary:"
        summary_output = llm.generate(summary_prompt, max_tokens=250, temp=0.3)
        interview_summary = summary_output.replace("Summary:", "").strip()
        if not interview_summary:
            interview_summary = "You did a solid job overall, showing a strong grasp of core textbook concepts. To reach the next level, practice articulating answers with more detail and clear structural points."
            
        # Save to session state
        st.session_state.interview_results = results
        st.session_state.interview_average = avg_score
        st.session_state.interview_summary = interview_summary
        
        # Navigate back to interview page
        st.session_state.current_page = "interview"
        st.rerun()
        
    except Exception as err:
        st.error(f"❌ Error during interview grading: {err}")
        # Recover to setup
        if st.button("Return to Setup"):
            st.session_state.current_page = "interview"
            st.session_state.interview_active = False
            st.rerun()
    finally:
        progress_bar.empty()
        status_text.empty()


elif st.session_state.current_page == "summarize":
    col_back, col_title = st.columns([1, 4])
    with col_back:
        st.write("")
        st.write("")
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.title("📝 Academic Document Summarizer")
        st.write("Upload a research paper, note, or textbook PDF/TXT to get an instant summary and ask document-specific questions.")

    st.markdown("---")

    uploaded_file = st.file_uploader("Upload Academic Document", type=["pdf", "txt"])

    # Parse and store document text automatically on upload
    if uploaded_file:
        if st.session_state.summary_file_name != uploaded_file.name or not st.session_state.summarizer_doc_text:
            st.session_state.summary_file_name = uploaded_file.name
            st.session_state.summary_text = ""
            st.session_state.summarizer_doc_text = ""
            st.session_state.summarizer_chunks = []
            st.session_state.summarizer_embeddings = []
            st.session_state.summarizer_chat_history = []
            
            with st.spinner("📖 Loading document..."):
                try:
                    # Check if precomputed embeddings exist
                    precomputed_path = os.path.join("embeddings", uploaded_file.name.rsplit(".", 1)[0] + ".json")
                    if os.path.exists(precomputed_path):
                        with open(precomputed_path, "r", encoding="utf-8") as f:
                            cached_data = json.load(f)
                        st.session_state.summarizer_chunks = [c.get("text", "") for c in cached_data]
                        st.session_state.summarizer_embeddings = np.array([c.get("embedding", []) for c in cached_data], dtype=np.float32)
                        st.session_state.summarizer_doc_text = "\n\n".join(st.session_state.summarizer_chunks)
                    else:
                        raw_text = ""
                        if uploaded_file.name.endswith(".pdf"):
                            uploaded_file.seek(0)
                            reader = pypdf.PdfReader(uploaded_file)
                            for page in reader.pages:
                                raw_text += page.extract_text() + "\n"
                            
                            # Fallback to PyTorch OCR if normal text extraction yields nothing
                            if len(raw_text.strip()) < 50:
                                uploaded_file.seek(0)
                                file_bytes = uploaded_file.read()
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
                        else:
                            uploaded_file.seek(0)
                            raw_text = str(uploaded_file.read(), "utf-8")
                        
                        st.session_state.summarizer_doc_text = raw_text
                        
                        # In-memory chunking and indexing
                        chunks = chunk_text(raw_text)
                        st.session_state.summarizer_chunks = chunks
                        
                        if chunks:
                            # Use the globally cached embedder to compute chunk embeddings
                            embeddings = embedder.encode(chunks, convert_to_numpy=True)
                            st.session_state.summarizer_embeddings = embeddings
                        else:
                            st.session_state.summarizer_embeddings = []
                except Exception as e:
                    st.error(f"❌ Error parsing file: {e}")


    # --- Document Q&A Chat Box Section ---
    if uploaded_file and st.session_state.summarizer_doc_text:
        st.markdown("---")
        st.subheader("💬 Ask Questions About the Document")
        st.write("Ask follow-up questions or prompt the assistant (e.g. *'just summarize the introduction'*).")

        # Display Chat History
        for msg in st.session_state.summarizer_chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        # Chat Input
        if user_query := st.chat_input("Ask a question about the uploaded document..."):
            # Save user message to history
            st.session_state.summarizer_chat_history.append({"role": "user", "content": user_query})
            
            # Save assistant response to history
            try:
                chunks = st.session_state.get("summarizer_chunks", [])
                embeddings = st.session_state.get("summarizer_embeddings", [])
                
                if chunks and len(embeddings) > 0:
                    # Embed user query
                    query_emb = embedder.encode(user_query, convert_to_numpy=True)
                    
                    # Compute cosine similarity
                    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_emb)
                    norms[norms == 0] = 1e-9
                    similarities = np.dot(embeddings, query_emb) / norms
                    
                    # Get top 3 chunks
                    top_indices = np.argsort(similarities)[::-1][:3]
                    
                    retrieved_chunks = []
                    # Check if query is about abstract or introduction
                    lowered_query = user_query.lower()
                    if "abstract" in lowered_query or "introduction" in lowered_query or "summary" in lowered_query:
                        if 0 not in top_indices:
                            retrieved_chunks.append(chunks[0])
                    
                    for idx in top_indices:
                        retrieved_chunks.append(chunks[idx])
                        
                    context = "\n\n... [Section Boundary] ...\n\n".join(retrieved_chunks)
                else:
                    context = st.session_state.summarizer_doc_text
                
                if len(context) > 6000:
                    context = context[:6000]
                
                prompt = f"""<|system|>
You are a precise academic assistant. Answer the user's question or follow their instruction based only on the provided document text. Keep your response concise, accurate, and direct. If the answer cannot be found in the text, state that. Do not add conversational filler.
Document Text:
{context}
</s>
<|user|>
Question/Instruction: {user_query}
</s>
<|assistant|>
Response:"""
                response_output = llm.generate(prompt, max_tokens=500, temp=0.2)
                clean_response = response_output.replace("Response:", "").strip()
                
                if not clean_response:
                    clean_response = "Could not generate a response."
                    
                # Save assistant response to history
                st.session_state.summarizer_chat_history.append({"role": "assistant", "content": clean_response})
            except Exception as err:
                st.error(f"Error querying AI: {err}")
            st.rerun()

import urllib.parse
import time

def skill_analysis_agent(user_input):
    """
    Skill Analysis Agent:
    Extracts and normalizes skills from the user input.
    """
    if not user_input.strip():
        return []
    parts = re.split(r'[,;]+', user_input)
    skills = [p.strip().title() for p in parts if p.strip()]
    return list(set(skills))

def career_recommendation_agent(skills):
    """
    Career Recommendation Agent:
    Recommends career paths based on the list of skills.
    """
    if not skills:
        return []
    skills_str = ", ".join(skills)
    prompt = f"""<|system|>
You are a helpful Career Path Advisor. Based on the technical skills provided, recommend exactly 3 suitable academic or professional career tracks. Output only the career names as a simple list (one per line). Do not add any introduction, explanations, numbering, or formatting.</s>
<|user|>
Skills: {skills_str}
</s>
<|assistant|>
"""
    output = llm.generate(prompt, max_tokens=150, temp=0.2).strip()
    lines = [line.strip().replace("-", "").replace("*", "").strip() for line in output.split("\n") if line.strip()]
    return [l for l in lines if len(l) > 3][:3]

def job_recommendation_agent(skills):
    """
    Job Recommendation Agent:
    Recommends specific job roles based on skills.
    """
    if not skills:
        return []
    skills_str = ", ".join(skills)
    prompt = f"""<|system|>
You are a helpful Recruitment Advisor. Based on the technical skills provided, recommend exactly 3 specific job roles. Output only the job role names as a simple list (one per line). Do not add any introduction, explanations, numbering, or formatting.</s>
<|user|>
Skills: {skills_str}
</s>
<|assistant|>
"""
    output = llm.generate(prompt, max_tokens=150, temp=0.2).strip()
    lines = [line.strip().replace("-", "").replace("*", "").strip() for line in output.split("\n") if line.strip()]
    if not lines:
        lines = [f"{skills[0]} Developer", "Software Engineer", "Systems Analyst"]
    return [l for l in lines if len(l) > 3][:3]

def linkedin_link_agent(job_roles):
    """
    LinkedIn Link Agent:
    Generates clickable LinkedIn job search URLs for each recommended job role.
    """
    links = {}
    for role in job_roles:
        encoded_role = urllib.parse.quote(role)
        links[role] = f"https://www.linkedin.com/jobs/search/?keywords={encoded_role}"
    return links

def interview_preparation_agent(selected_role):
    """
    Interview Preparation Agent:
    Generates 5 interview questions for the selected role using the local LLM.
    """
    prompt = f"""<|system|>
You are an expert Technical Interviewer. Generate exactly 5 challenging technical interview questions for the role specified.
Output only the numbered questions (1 to 5) with no introductory or concluding text. Do not provide answers.</s>
<|user|>
Job Role: {selected_role}
</s>
<|assistant|>
"""
    output = llm.generate(prompt, max_tokens=300, temp=0.3).strip()
    questions = [q.strip() for q in output.split("\n") if q.strip()]
    return questions[:5]

if st.session_state.current_page == "career":
    col_back, col_title = st.columns([1, 4])
    with col_back:
        st.write("")
        st.write("")
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.current_page = "main"
            st.rerun()
    with col_title:
        st.title("💼 AI Career Assistant")
        st.write("Specialized multi-agent workflow analyzing skills, recommending careers, jobs, and preparing you for interviews.")

    st.markdown("---")

    # 1. User Skills Input
    skills_input = st.text_input(
        "Enter your Skills (comma-separated):",
        placeholder="e.g. Python, SQL, Machine Learning",
        key="career_skills_text_input"
    )

    if st.button("🚀 Run Career Analysis Pipeline", type="primary", use_container_width=True):
        if not skills_input.strip():
            st.warning("⚠️ Please enter some skills first.")
        else:
            status_text = st.empty()
            progress_bar = st.progress(0.0)

            # Step 2: Skill Analysis
            status_text.text("🔍 Analyzing and normalizing skills...")
            progress_bar.progress(0.2)
            analyzed_skills = skill_analysis_agent(skills_input)
            st.session_state.career_analyzed_skills = analyzed_skills

            # Step 3: Career Recommendation
            status_text.text("📈 Recommending career paths...")
            progress_bar.progress(0.4)
            careers = career_recommendation_agent(analyzed_skills)
            if not careers:
                careers = ["Data Science Track", "Software Engineering Track", "Data Analyst Track"]
            st.session_state.career_paths = careers

            # Step 4: Job Recommendation
            status_text.text("💼 Recommending specific job roles...")
            progress_bar.progress(0.6)
            jobs = job_recommendation_agent(analyzed_skills)
            st.session_state.career_jobs = jobs

            # Step 5: LinkedIn links
            status_text.text("🔗 Generating LinkedIn job search links...")
            progress_bar.progress(0.8)
            links = linkedin_link_agent(jobs)
            st.session_state.career_links = links

            progress_bar.progress(1.0)
            status_text.text("✅ Analysis complete!")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            st.rerun()

    # Render results if we have analyzed skills
    if st.session_state.career_analyzed_skills:
        st.markdown("---")
        
        # Display detected skills as badges
        st.subheader("🔍 Detected Skills")
        badges_html = " ".join([f'<span class="score-badge" style="margin-right: 8px; font-size: 1.0em; padding: 5px 12px;">{s}</span>' for s in st.session_state.career_analyzed_skills])
        st.markdown(f'<div style="margin-bottom: 20px;">{badges_html}</div>', unsafe_allow_html=True)

        col_c, col_j = st.columns(2)
        
        with col_c:
            st.subheader("📈 Recommended Career Paths")
            for path in st.session_state.career_paths:
                st.markdown(f"- **{path}**")
                
        with col_j:
            st.subheader("💼 Recommended Job Roles")
            for job in st.session_state.career_jobs:
                st.markdown(f"- **{job}**")

        st.markdown("---")
        st.subheader("🔗 LinkedIn Job Search Links")
        st.write("Click below to search for live openings matching the recommended roles:")
        for job, url in st.session_state.career_links.items():
            st.markdown(f"- [Search for **{job}** on LinkedIn 🚀]({url})")

        st.markdown("---")
        st.subheader("🎙️ Interview Preparation Agent")
        st.write("Select one of the recommended roles to generate mock interview questions:")
        
        selected_role = st.selectbox(
            "Select Role:",
            options=st.session_state.career_jobs,
            key="selected_role_dropdown"
        )
        
        if st.button("🧠 Generate Interview Questions", use_container_width=True):
            with st.spinner("🤖 Generating questions using TinyLlama..."):
                questions = interview_preparation_agent(selected_role)
                st.session_state.career_interview_questions = questions
                st.session_state.selected_career_role = selected_role
                st.rerun()

        if st.session_state.career_interview_questions and st.session_state.selected_career_role == selected_role:
            st.markdown(f"#### 📝 Questions for {selected_role}")
            for i, q in enumerate(st.session_state.career_interview_questions):
                st.markdown(f"**{i+1}.** {q}")
