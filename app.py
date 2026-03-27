from PIL import Image
import torch
import clip

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
    return text_features

def encode_image(image_path, model, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = model.encode_image(image)
    return image_features
    
def retrieve(image_features, text_features, docs):
    similarities = (image_features @ text_features.T).squeeze(0)
    best_idx = similarities.argmax().item()
    return docs[best_idx], similarities[best_idx].item()

def main():
    model, preprocess, device = load_model()

    docs = load_documents("docs/data.txt")
    text_features = encode_texts(docs, model, device)
    image_path = "images/test.jpg"    
    image_features = encode_image(image_path, model , preprocess, device)

    best_doc, score = retrieve(image_features, text_features, docs)
    print("Most relevant documents: ")
    print(best_doc)
    print(f"\nSimilarity Score: {score:.4f}")


if __name__ == "__main__":
    main()