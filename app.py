import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
from PIL import Image
import torch
import clip
import faiss
from openai import OpenAI

FAISS_CANDIDATES = 8 # How many raw docs to fetch first
FINAL_TOP_K = 3 # How many final docs to show
ALPHA = 0.85 # Weight for semantic score
BETA = 0.15 # Weight for lexical score

def build_faiss_index(text_features):
    # Convert to numpy
    vectors = text_features.cpu().numpy()

    dim = vectors.shape[1] # embedding size(512)

    index = faiss.IndexFlatL2(dim)
    index.add(vectors)

    return index

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device = device)
    return model, preprocess, device

def load_documents(path):
    with open(path, "r", encoding="utf-8") as f:
        docs = [chunk.strip() for chunk in f.read().split("\n\n") if chunk.strip()]
    return docs

def encode_texts(docs, model, device):
    text_tokens = clip.tokenize(docs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim = True)
    return text_features

def encode_image(image_path, model, preprocess, device):
    # For streamlit
    if isinstance(image_path, str):
        image = Image.open(image_path)
    else:
        image = image_path
    image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim = True)
    return image_features
    
def clean_text(text):
    text = text.lower()
    for ch in "[](),.:;!?-_/\n":
        text = text.replace(ch, " ")
    return text

def lexical_score(doc, candidate_docs):
    """
    Penalizes generic documents.
    More unique words → higher score.
    More common words → lower score.
    """
    doc_words = set(clean_text(doc).split())

    word_counts = {}

    # Count word frequency across candidates
    for candidate in candidate_docs:
        words = set(clean_text(candidate).split())
        for w in words:
            word_counts[w] = word_counts.get(w, 0) + 1

    score = 0.0
    for w in doc_words:
        freq = word_counts.get(w, 1)
        score += 1 / freq   # rare words get higher weight

    return score / len(doc_words) if doc_words else 0.0

def retrieve_faiss(image_features, index, docs, k=FINAL_TOP_K):
    query = image_features.cpu().numpy()

    # Step 1: get more candidates from FAISS
    distances, indices = index.search(query, k=FAISS_CANDIDATES)

    candidates = []
    candidate_docs = []

    for i in range(FAISS_CANDIDATES):
        idx = indices[0][i]
        distance = distances[0][i]
        doc = docs[idx]

        candidates.append((doc, distance))
        candidate_docs.append(doc)

    # Step 2: compute hybrid scores
    hybrid_results = []

    for doc, distance in candidates:
        semantic_score = 1 / (1 + distance)   # convert lower distance to higher score
        lexical = lexical_score(doc, candidate_docs)

        combined_score = ALPHA * semantic_score + BETA * lexical

        hybrid_results.append((doc, distance, semantic_score, lexical, combined_score))

    # Step 3: rerank by combined score
    hybrid_results.sort(key=lambda x: x[4], reverse=True)

    # Step 4: keep top-k
    final_results = [(doc, distance, combined_score) for doc, distance, _, _, combined_score in hybrid_results[:k]]

    # confidence still depends on best FAISS distance
    best_score = hybrid_results[0][4]  # combined_score of top result
    
    # thresholds empirically tuned based on observed hybrid scores
    # for relevant vs irrelevant retrieval cases
    if best_score > 0.47:
        confidence_level = "high"
    elif best_score > 0.44:
        confidence_level = "medium"
    else:
        confidence_level = "low"
        
    return final_results, confidence_level

def generate_explanation(results, confidence_level):
    if confidence_level == "low":
        return "The system could not find a confident match for this image based on the available knowledge base."
    if confidence_level == "medium":
        prefix = "The following explanation is moderately confident based on the retrieved knowledge.\n\n"
    else:
        prefix = ""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return(
            "LLM skipped (no API key)" 
        )
    client = OpenAI(api_key=api_key)
    context =""
    for i, (doc, distance, combined_score) in enumerate(results[:2]):
        context += (
        f"\nDocument {i+1} "
        f"(distance {distance:.4f}, hybrid score {combined_score:.4f}):\n{doc}\n"
    )
    prompt = f"""
You are analyzing an image using retrieved knowledge.

The system found the following relevant information:
{context}

Write a clear and confident explanation of what is in the image:
- describe the scene directly and naturally (avoid "may", "might" where possible)
- combine the retrieved information logically
- avoid vague phrases like "present in the scene"
- do not mention document numbers
- mention that lower distance means higher similarity
- mention that higher hybrid score means stronger overall relevance
- keep it natural and concise
"""
    try:
        response = client.responses.create(
            model="gpt-5.4-mini",
            input=prompt
        )
        return prefix + response.output_text
    except Exception as e:
        return f"LLM explanation failed: {e}"

def main():
    print("Loading model..")
    model, preprocess, device = load_model()

    print("Loading documents..")
    docs = load_documents("docs/data.txt")

    print("Encoding text..")
    text_features = encode_texts(docs, model, device)

    print("Encoding image..")
    image_path = "images/test2.jpg"    
    image_features = encode_image(image_path, model , preprocess, device)

    # best_doc, score = retrieve(image_features, text_features, docs)
    # build index
    print("Building FAISS index..")
    index = build_faiss_index(text_features)

    print("Retrieving..")
    results, confidence_level = retrieve_faiss(image_features, index, docs, k=3)
    print(f"\nConfidence Level: {confidence_level.upper()}")
    print("\n Top Retrieved Documents:")
    
    for i, (doc, distance, combined_score) in enumerate(results):
        print(f"\nRank {i+1}:")
        print(doc)
        print(f"Distance: {distance:.4f} | Hybrid Score: {combined_score:.4f}")

    print("\nGenerating explanation:")
    explanation = generate_explanation(results, confidence_level)

    print("\nLLM Explanation:")
    print(explanation)

if __name__ == "__main__":
    main()

