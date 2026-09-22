# AI-Powered Academic Assistant

An AI-powered academic assistance platform that helps students understand academic documents, generate context-aware answers, summarize documents, evaluate answers, prepare for interviews, and receive career guidance.

The system combines **Retrieval-Augmented Generation (RAG)**, **Sentence Transformers**, **FAISS**, and **TinyLlama 1.1B** to provide document-grounded responses. It also includes an AI Career Assistant that performs skill analysis, career recommendations, job recommendations, LinkedIn job search, and interview preparation.

---

## 📌 Project Overview

Students and researchers often work with large academic documents such as research papers, textbooks, and study materials. Finding relevant information manually can be time-consuming, and traditional search systems may not provide context-aware answers.

The **AI-Powered Academic Assistant** provides a unified platform where users can upload academic documents and interact with them using AI.

The system uses **Retrieval-Augmented Generation (RAG)** to retrieve relevant information from uploaded documents before generating an answer. This helps the language model generate responses based on the available academic context instead of relying only on its internal knowledge.

The platform also provides document summarization, answer evaluation, interview preparation, and career assistance.

---

## 🎯 Objectives

- Develop an AI-powered platform for academic assistance.
- Provide context-aware question answering from uploaded academic documents.
- Implement Retrieval-Augmented Generation (RAG).
- Use Sentence Transformers for semantic embedding generation.
- Use FAISS for efficient similarity-based document retrieval.
- Use TinyLlama 1.1B for response generation.
- Provide document summarization.
- Evaluate descriptive answers using multiple evaluation techniques.
- Generate customized technical interview questions.
- Provide skill-based career and job recommendations.
- Provide LinkedIn job search links.
- Create an integrated academic and career support system for students.

---

## ✨ Key Features

### 1. 📚 Document-Based Question Answering

Users can upload academic documents and ask questions related to the uploaded content.

The system:

1. Extracts text from the document.
2. Splits the document into meaningful chunks.
3. Generates embeddings for the chunks.
4. Stores the embeddings in FAISS.
5. Converts the user's question into an embedding.
6. Retrieves the most relevant chunks.
7. Provides the retrieved context to TinyLlama.
8. Generates a context-aware answer.

---

### 2. 📝 Document Summarization

The system allows users to summarize specific sections of academic documents.

Users can select sections such as:

- Abstract
- Introduction
- Methodology
- Results
- Conclusion

The selected content is processed using the local language model to generate a concise summary.

---

### 3. 📊 Answer Evaluation

The system evaluates student answers against reference answers using multiple evaluation techniques.

The evaluation module uses:

- **Cosine Similarity**
- **Keyword Matching**
- **BERTScore**

These methods provide different perspectives on answer similarity.

**Cosine Similarity** evaluates overall semantic similarity.

**Keyword Matching** checks whether important concepts or keywords are present.

**BERTScore** compares contextual similarity between the candidate answer and the reference answer.

Using multiple metrics helps provide a more comprehensive evaluation of descriptive answers.

---

### 4. 🎤 Interview Assistant

The Interview Assistant generates customized technical interview questions.

Users can specify:

- Technical area
- Difficulty/complexity
- Number of questions

The system uses the available context and local language model to generate interview questions suitable for the selected area.

---

### 5. 💼 AI Career Assistant

The AI Career Assistant provides career guidance based on the technical skills entered by the user.

The module follows a multi-agent workflow consisting of:

#### Skill Analysis Agent

Processes and normalizes the user's technical skills.

Example:

```text
Input:
python, machine learning, sql

Output:
Python
Machine Learning
SQL
