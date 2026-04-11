import streamlit as st
from PIL import Image
import requests
import os

st.set_page_config(page_title = "Multimodal Image Understanding with RAG", layout = "wide")

st.title("Multimodal Image Understanding with RAG")
st.write("Upload an image and get a grounded explanation using Retrieval+LLM")


API_URL = os.getenv("API_URL", "http://127.0.0.1:8000/analyze")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    
    # Show Image
    image = Image.open(uploaded_file)
    # Layout Creation
    col1, col2 = st.columns(2)
    
    # Show the image
    with col1:
        st.image(image, caption="Uploaded Image", width=800)
    # Process after Display 
    with st.spinner("Analyzing image using CLIP + retrieval + LLM..."):
        try:
            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    uploaded_file.type
                )
            }

            response = requests.post(API_URL, files=files, timeout=30)

            if response.status_code != 200:
                st.error(f"API Error: {response.status_code}")
                st.stop()

            data = response.json()


        except requests.exceptions.Timeout:
            st.error("Request timed out. Try again.")
            st.stop()
            
        except Exception as e:
            st.error(f"Failed to connect to API: {e}")
            st.stop()
    
    with col2:

        # Confidence
        confidence = data["confidence"]

        if confidence == "high":
            st.success("High confidence match")
        elif confidence == "medium":
            st.warning("Moderate confidence match")
        else:
            st.error("Low confidence match")
        
        if "latency" in data:
            st.caption(f"Latency: {data['latency']} seconds")

        # Explanation
        st.subheader("Explanation")
        st.write(data["explanation"])

        st.divider()

        # Supporting Knowledge
        st.subheader("Supporting Knowledge")

        if confidence == "low":
            st.info("No reliable supporting knowledge found.")
        else:
            for i, item in enumerate(data["results"]):
                doc = item["doc"]
                distance = item["distance"]
                score = item["score"]

                if "]" in doc:
                    tag = doc.split("]")[0] + "]"
                    content = doc.split("]")[1]
                else:
                    tag = "[INFO]"
                    content = doc

                st.markdown(
                    f"**Rank {i+1} — {tag} "
                    f"(Distance: {distance:.4f}, Hybrid Score: {score:.4f})**"
                )

                st.write(content.strip())
        