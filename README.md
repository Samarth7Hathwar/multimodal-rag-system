# Multimodal RAG System

## 🚀 Overview
This project builds a multimodal AI system that:
- Understands images using CLIP
- Retrieves relevant documents using embeddings
- (Upcoming) Generates explanations using LLMs

---

## 🧠 Current Progress

### ✅ Completed
- Image loading with PIL
- Zero-shot image understanding (CLIP)
- Basic multimodal retrieval (image → relevant text)

---

## 🔍 Example Output

Input: Image of a child with a camel

Output: Most Relevant Document:
Children often express emotions like happiness and laughter when playing.

Similarity Score: 34.8192

---

## 🏗️ Architecture
Image → CLIP → Image Embedding
Text → CLIP → Text Embeddings
→ Similarity Search (Dot Product)
→ Best Matching Document

---

## 🛠️ Tech Stack

- Python
- PyTorch
- CLIP
- PIL

Upcoming:
- FAISS (fast retrieval)
- LLM (explanations)
- Streamlit UI

---

## 📂 Project Structure
images/
docs/
src/
app.py
README.md
requirements.txt

---

## 🎯 Next Steps

- [ ] Replace brute-force search with FAISS
- [ ] Add LLM reasoning layer
- [ ] Build UI