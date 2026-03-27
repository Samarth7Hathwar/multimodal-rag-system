# Multimodal RAG System

## 🚀 Overview
This project builds a **Multimodal AI system** that can:
- Understand images using CLIP (vision-language model)
- Convert both images and text into embeddings
- Retrieve the most relevant documents using FAISS (vector search)

The goal is to evolve this into a full **Retrieval-Augmented Generation (RAG)** system that explains images using retrieved knowledge.

---

## 🧠 Current Progress

### ✅ Completed
- Image loading and preprocessing (PIL)
- Zero-shot image understanding using CLIP
- Text embedding generation using CLIP
- Multimodal retrieval (image → relevant text)
- FAISS-based vector search (efficient retrieval)
- Embedding normalization for accurate similarity comparison

---

## 🔍 Example Output

**Input:** Image of a child with a camel  

**Output:**

Most relevant document:
Children often express emotions like happiness and laughter when playing.

Distance Score (lower is better): 1.4075

---

## 🏗️ System Architecture
Image → CLIP → Image Embedding
Text → CLIP → Text Embeddings
↓
FAISS Index
↓
Nearest Neighbor Search
↓
Most Relevant Document

---

## 🛠️ Tech Stack

- Python
- PyTorch
- CLIP (Vision-Language Model)
- FAISS (Vector Similarity Search)
- PIL (Image Processing)

---

## 📂 Project Structure
multimodal-rag-system/
│── images/
│── docs/
│── src/
│── app.py
│── README.md
│── requirements.txt

---

## 🧠 Key Concepts Implemented

- Embeddings (image + text)
- Vector similarity (dot product & L2 distance)
- Embedding normalization
- Efficient nearest neighbor search (FAISS)

---

## 🎯 Next Steps

- [ ] Add LLM-based explanation layer
- [ ] Combine retrieved context with reasoning
- [ ] Build API using FastAPI
- [ ] Create UI using Streamlit
- [ ] Improve dataset (real-world documents)

---

## ⚠️ Note

This project is being built step-by-step to understand how real-world AI systems work, focusing on **core concepts first (embeddings, retrieval, indexing)** before adding higher-level components like LLMs.