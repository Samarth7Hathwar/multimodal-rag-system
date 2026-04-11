from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import time

from app import (
    load_model,
    load_documents,
    encode_texts,
    encode_image,
    build_faiss_index,
    retrieve_faiss,
    generate_explanation
)

app = FastAPI()

# Load once at startup
model, preprocess, device = load_model()
docs = load_documents("docs/data.txt")
if len(docs) == 0:
    raise ValueError("No documents found in docs/data.txt")

text_features = encode_texts(docs, model, device)
index = build_faiss_index(text_features)


@app.get("/")
def root():
    return {"message": "Multimodal RAG API is running"}

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    start_time = time.time()

    # Read image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Encode image
    image_features = encode_image(image, model, preprocess, device)

    # Retrieve knowledge
    results, confidence_level = retrieve_faiss(
        image_features,
        index,
        docs,
        k=3
    )

    # Generate explanation (from app.py)
    explanation = generate_explanation(results, confidence_level)

    latency = round(time.time() - start_time, 3)

    # Format response
    response = {
        "confidence": confidence_level,
        "latency": latency,
        "results": [
            {
                "doc": doc,
                "distance": float(distance),
                "score": float(combined_score)
            }
            for doc, distance, combined_score in results
        ],
        "explanation": explanation
    }

    return response