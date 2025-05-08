import streamlit as st
from PIL import Image
import time
import random

st.set_page_config(layout="wide")

left_col, right_col = st.columns(2)


def use_bi_encoder():
    """
    Here need to call the model and then do the following: 
        
        result_img = Image type

        st.session_state["result_img"] = result_img
    """
    pass

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
