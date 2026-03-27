from PIL import Image
import torch
import clip
import faiss
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

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
    with open(path,"r") as f:
        docs = f.read().split("\n\n")
    return docs

def encode_texts(docs, model, device):
    text_tokens = clip.tokenize(docs).to(device)
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim = True)
    return text_features

def encode_image(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=-1, keepdim = True)
    return image_features
    
def retrieve_faiss(image_features, index, docs):
    query = image_features.cpu().numpy()

    D, I = index.search(query, k=1) # k=1 -> best match

    best_idx = I[0][0]
    score = D[0][0]

    return docs[best_idx], score

def main():
    print("Loading model..")
    model, preprocess, device = load_model()

    print("Loading documents..")
    docs = load_documents("docs/data.txt")

    print("Encoding text..")
    text_features = encode_texts(docs, model, device)

    print("Encoding image..")
    image_path = "images/test.jpg"    
    image_features = encode_image(image_path, model , preprocess, device)

    # best_doc, score = retrieve(image_features, text_features, docs)
    # build index
    print("Building FAISS index..")
    index = build_faiss_index(text_features)

    print("Retrieving..")
    best_doc, score = retrieve_faiss(image_features, index , docs)

    print("Most relevant document: ")
    print(best_doc)
    #print(f"\nSimilarity Score: {score:.4f}")
    print(f"\nDistance Score (lower is better): {score:.4f}")


if __name__ == "__main__":
    main()

