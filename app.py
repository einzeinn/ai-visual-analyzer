import streamlit as st
import cv2
import numpy as np
from PIL import Image
from detector import detect_objects
from collections import Counter
import pandas as pd

st.title("AI Visual Analyzer")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.5)


if uploaded_file is not None:
    
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    
    st.image(image, caption="Uploaded Image")
    

    if st.button("Analyze Image"):

        result_img, detections = detect_objects(image_np, threshold)

        result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        
        st.image(result_img, caption="Detection Result")

        # convert image supaya bisa didownload
        _, buffer = cv2.imencode(".png", result_img)

        st.download_button(
            label="Download Detection Result",
            data=buffer.tobytes(),
            file_name="detection_result.png",
            mime="image/png"
        )   

        st.subheader("Detected Objects")

        labels = [obj["label"] for obj in detections]
        counts = Counter(labels)

        for label, count in counts.items():
         st.write(f"{label}: {count}")

        for obj in detections:
            st.write(f"{obj['label']} — confidence: {obj['confidence']}%")
        st.subheader("Detection Table")

        df = pd.DataFrame(detections)

        st.dataframe(df)


st.subheader("Object Count")

