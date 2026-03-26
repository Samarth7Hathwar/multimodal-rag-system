from PIL import Image
import torch
import clip

def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device = device)
    return model, preprocess, device

def process_image(image_path, preprocess, device):
    image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
    return image


def main():
    image_path = "images/test.jpg"
    #image = load_image(image_path)

    print("Loading CLIP model..")
    model, preprocess, device = load_model()

    print("Preprocessing image..")
    image = process_image(image_path, preprocess,device)

    labels = [
        "a child",
        "a camel",
        "a desert",
        "a person laughing",
        "an animal"
    ]
    text = clip.tokenize(labels).to(device)

    with torch.no_grad():
        logits_per_image, _ = model(image,text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    print("\n Predictions:")
    for labels, prob in zip(labels, probs[0]):
        print(f"{labels}: {prob:.4f}")


if __name__ == "__main__":
    main()