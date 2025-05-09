import streamlit as st
from PIL import Image
import time
import random
import torch
import numpy as np
import cv2

import torch
import cv2
import numpy as np

from machine_learning_project.bi_encoder import BiEncoder
from machine_learning_project.dataset_utils import detect_keypoints_and_descriptors
from machine_learning_project.poly_encoder import PolyEncoder
from machine_learning_project.lightGlue import LightGlue

st.set_page_config(layout="wide")

left_col, right_col = st.columns(2)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def use_bi_encoder(img1, img2):
    img1 = cv2.imdecode(np.frombuffer(img1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(img2.read(), np.uint8), cv2.IMREAD_COLOR)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder1 = BiEncoder().to(device)
    encoder2 = BiEncoder().to(device)
    checkpoint = torch.load("bi_encoder.pth", map_location=device)
    encoder1.load_state_dict(checkpoint['encoder1_state_dict'])
    encoder2.load_state_dict(checkpoint['encoder2_state_dict'])

    data1 = detect_keypoints_and_descriptors(img1, detector_type="ORB", n_features=512)
    data2 = detect_keypoints_and_descriptors(img2, detector_type="ORB", n_features=512)

    descriptors1 = data1["descriptors"]
    descriptors2 = data2["descriptors"]
    keypoints1 = np.array([kp.pt for kp in data1["keypoints"]], dtype=np.float32)
    keypoints2 = np.array([kp.pt for kp in data2["keypoints"]], dtype=np.float32)
    
    if len(keypoints1) == 0 or len(keypoints2) == 0:
        st.warning("Could not extract features from one of the images.")
        return

    input1 = torch.tensor(np.hstack([keypoints1, descriptors1]), dtype=torch.float32).to(device)
    input2 = torch.tensor(np.hstack([keypoints2, descriptors2]), dtype=torch.float32).to(device)
    with torch.no_grad():
        emb1 = encoder1(input1)
        emb2 = encoder2(input2)

    emb1_norm = emb1 / emb1.norm(dim=1, keepdim=True)
    emb2_norm = emb2 / emb2.norm(dim=1, keepdim=True)
    sim = emb1_norm @ emb2_norm.T  # (N1, N2)
    matches = torch.argmax(sim, dim=1).cpu().numpy()
    # max_sims = torch.max(sim, dim=1).values.cpu().numpy()

    # kp1_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints1]
    # kp2_cv = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints2]
    # dmatches = [cv2.DMatch(_queryIdx=i, _trainIdx=int(matches[i]), _distance=float(1 - max_sims[i])) for i in range(len(matches))]

    # matched_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, dmatches, None)

    # matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    # result_pil = Image.fromarray(matched_img)

    max_sims, indices = torch.max(sim, dim=1)
    max_sims_np = max_sims.cpu().numpy()
    indices_np = indices.cpu().numpy()

    topk = 100
    if len(max_sims_np) < topk:
        topk = len(max_sims_np)

    topk_indices = np.argsort(max_sims_np)[-topk:]

    kp1_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints1]
    kp2_cv = [cv2.KeyPoint(float(x), float(y), 1) for x, y in keypoints2]

    dmatches = [
        cv2.DMatch(_queryIdx=int(i), _trainIdx=int(indices_np[i]), _distance=float(1 - max_sims_np[i]))
        for i in topk_indices
    ]

    matched_img = cv2.drawMatches(img1, kp1_cv, img2, kp2_cv, dmatches, None)

    matched_img = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)
    result_pil = Image.fromarray(matched_img)

    st.session_state["result_img"] = result_pil

    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2

    # Draw matched keypoints
    for (x0, y0) in keypoints1:
        cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 255, 0), -1)  

    for (x1, y1) in keypoints2:
        cv2.circle(canvas, (int(x1) + w1, int(y1)), 2, (0, 255, 0), -1)  

    keypoints_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    st.session_state["keypoints_img"] = keypoints_pil



def use_light_glue(img1, img2):
    desc_dim = 32
    model_path = "/home/ironiss/Documents/ML-project/machine_learning_project/best_lightglue_encoder.pth"
    detector_type = "ORB"
    threshold = 0.35

    model = LightGlue(
            input_dim=2+desc_dim,
            hidden_dim=256,
            gnn_layers=4
        )

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()


    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    data1 = detect_keypoints_and_descriptors(img1, detector_type=detector_type, n_features=512)
    data2 = detect_keypoints_and_descriptors(img2, detector_type=detector_type, n_features=512)

    keypoints1 = np.array([kp.pt for kp in data1["keypoints"]], dtype=np.float32)
    keypoints2 = np.array([kp.pt for kp in data2["keypoints"]], dtype=np.float32)

    descriptors1 = data1["descriptors"]
    descriptors2 = data2["descriptors"]

    feat1 = torch.tensor(np.hstack([keypoints1, descriptors1]), dtype=torch.float32).unsqueeze(0).to(device)
    feat2 = torch.tensor(np.hstack([keypoints2, descriptors2]), dtype=torch.float32).unsqueeze(0).to(device)


    with torch.no_grad():
        emb1, emb2 = model(feat1, feat2)

    similarity = torch.bmm(emb1, emb2.transpose(1, 2))[0] 
    pred_matches = torch.argmax(similarity, dim=1)         
    confidences = similarity[torch.arange(similarity.size(0)), pred_matches]


    kp1_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]
    kp2_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints2]


    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    st.session_state["result_img"] = matched_img_rgb

    matched_img = cv2.drawMatches(
        img1,
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1_cv],
        img1,
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp2_cv],
        [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1_cv))],
        None
    )

    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2

    for (x0, y0) in img1:
        cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 255, 0), -1)  

    for (x1, y1) in img2:
        cv2.circle(canvas, (int(x1) + w1, int(y1)), 2, (0, 255, 0), -1)  

    keypoints_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    st.session_state["keypoints_img"] = keypoints_pil

def use_poly_encoder(img1, img2):
    desc_dim = 32
    model_path = "/home/ironiss/Documents/ML-project/machine_learning_project/best_poly_encoder.pth"
    detector_type = "ORB"
    threshold = 0.35


    model = PolyEncoder(
        input_dim=2 + desc_dim,
        hidden_dim=256,
        num_heads=8,
        dropout=0.1
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()


    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    data1 = detect_keypoints_and_descriptors(img1, detector_type=detector_type, n_features=512)
    data2 = detect_keypoints_and_descriptors(img2, detector_type=detector_type, n_features=512)

    keypoints1 = np.array([kp.pt for kp in data1["keypoints"]], dtype=np.float32)
    keypoints2 = np.array([kp.pt for kp in data2["keypoints"]], dtype=np.float32)

    descriptors1 = data1["descriptors"]
    descriptors2 = data2["descriptors"]

    feat1 = torch.tensor(np.hstack([keypoints1, descriptors1]), dtype=torch.float32).unsqueeze(0).to(device)
    feat2 = torch.tensor(np.hstack([keypoints2, descriptors2]), dtype=torch.float32).unsqueeze(0).to(device)


    with torch.no_grad():
        emb1, emb2 = model(feat1, feat2)

    similarity = torch.bmm(emb1, emb2.transpose(1, 2))[0] 
    pred_matches = torch.argmax(similarity, dim=1)         
    confidences = similarity[torch.arange(similarity.size(0)), pred_matches]


    kp1_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints1]
    kp2_cv = [cv2.KeyPoint(float(pt[0]), float(pt[1]), 1) for pt in keypoints2]


    matched_img_rgb = cv2.cvtColor(matched_img, cv2.COLOR_BGR2RGB)

    st.session_state["result_img"] = matched_img_rgb

    matched_img = cv2.drawMatches(
        img1,
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp1_cv],
        img1,
        [cv2.KeyPoint(float(x), float(y), 1) for x, y in kp2_cv],
        [cv2.DMatch(_queryIdx=i, _trainIdx=i, _distance=0) for i in range(len(kp1_cv))],
        None
    )

    h = max(img1.shape[0], img2.shape[0])
    w1, w2 = img1.shape[1], img2.shape[1]
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:img1.shape[0], :w1] = img1
    canvas[:img2.shape[0], w1:w1 + w2] = img2

    for (x0, y0) in img1:
        cv2.circle(canvas, (int(x0), int(y0)), 2, (0, 255, 0), -1)  

    for (x1, y1) in img2:
        cv2.circle(canvas, (int(x1) + w1, int(y1)), 2, (0, 255, 0), -1)  

    keypoints_pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
    st.session_state["keypoints_img"] = keypoints_pil




with left_col:
    st.header("Input images")

    img1 = st.file_uploader("Load image 1: ", type=["jpg", "jpeg", "png"], key="img1")
    img2 = st.file_uploader("Load image 2: ", type=["jpg", "jpeg", "png"], key="img2")

    method = st.selectbox("Choose the method: ", ["Poly-Encoder", "Bi-encoder", "LightGlue"])

    if st.button("Submit"):
        if img1 is None or img2 is None:
            st.warning("Please load two images...")
        else:
            with st.spinner("Processing..."):
                if method == "Poly-Encoder":
                    use_poly_encoder(img1, img2)
                elif method == "Bi-encoder":
                    use_bi_encoder(img1, img2)
                else:
                    use_light_glue(img1, img2)

with right_col:
    st.header("Generated match")

    if "result_img" in st.session_state:
        st.image(st.session_state["result_img"], caption="Result Image", use_container_width=True)
        st.image(st.session_state["keypoints_img"], caption="Keypoints Image", use_container_width=True)
    else:
        st.info("Here will be your result image")