import streamlit as st
from PIL import Image
import time
import random
import torch
import numpy as np
import cv2

from machine_learning_project.bi_encoder import BiEncoder
from dataset_utils import extract_features

st.set_page_config(layout="wide")

left_col, right_col = st.columns(2)


def use_bi_encoder():
    """
    Here need to call the model and then do the following: 
        
        result_img = Image type

        st.session_state["result_img"] = result_img
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = BiEncoder().to(device)
    checkpoint = torch.load("bi_encoder.pth", map_location=device)
    encoder.load_state_dict(checkpoint['encoder1_state_dict'])
    encoder.eval()

    img1_data, keypoints1, descriptors1 = extract_features(st.session_state["img1"])
    img2_data, keypoints2, descriptors2 = extract_features(st.session_state["img2"])

    if len(keypoints1) == 0 or len(keypoints2) == 0:
        st.warning("Could not extract features from one of the images.")
        return

    # --- 3. Generate embeddings ---
    input1 = torch.tensor(np.hstack([keypoints1, descriptors1]), dtype=torch.float32).to(device)
    input2 = torch.tensor(np.hstack([keypoints2, descriptors2]), dtype=torch.float32).to(device)
    with torch.no_grad():
        emb1 = encoder(input1)
        emb2 = encoder(input2)

    # --- 4. Match using cosine similarity ---
    emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)
    sim = emb1_norm @ emb2_norm.T  # (N1, N2)
    matches = torch.argmax(sim, dim=1).cpu().numpy()
    max_sims = torch.max(sim, dim=1).values.cpu().numpy()

    # --- 5. Visualization ---
    kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints1]
    kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints2]
    dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=int(matches[i]), _distance=float(1 - max_sims[i])) for i in range(len(matches))]

    matched_img = cv2.drawMatches(img1_data, kp1_cv, img2_data, kp2_cv, dmatches, None)
    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(matched_img)

    # --- 6. Store result ---
    st.session_state["result_img"] = result_pil

def use_poly_encoder():
    pass


with left_col:
    st.header("Input images")

    img1 = st.file_uploader("Load image 1: ", type=["jpg", "jpeg", "png"], key="img1")
    img2 = st.file_uploader("Load image 2: ", type=["jpg", "jpeg", "png"], key="img2")

    method = st.selectbox("Choose the method: ", ["Poly-Encoder", "Bi-encoder"])

    if st.button("Submit"):
        if img1 is None or img2 is None:
            st.warning("Please load two images...")
        else:
            with st.spinner("Processing..."):

                if method == "Poly-Encoder":
                    use_poly_encoder()
                else:
                    use_bi_encoder()

                
with right_col:
    st.header("Generated match")

    if "result_img" in st.session_state:
        st.image(st.session_state["result_img"], caption="Result Image", use_container_width=True)
    else:
        st.info("Here will be your result image")
