import streamlit as st
from PIL import Image
import time
import random

st.set_page_config(layout="wide")

left_col, right_col = st.columns(2)

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
                time.sleep(2)  

                color = random.choice(["red", "green", "blue"])
                result_img = Image.new("RGB", (256, 256), color=color)

                st.session_state["result_img"] = result_img

with right_col:
    st.header("Generated match")

    if "result_img" in st.session_state:
        st.image(st.session_state["result_img"], caption="Result Image", use_container_width=True)
    else:
        st.info("Here will be your result image")
