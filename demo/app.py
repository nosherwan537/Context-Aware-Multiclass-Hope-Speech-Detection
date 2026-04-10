import streamlit as st

from src.normalizer import PhoneticNormalizer


st.set_page_config(page_title="HLSP Demo", layout="centered")
st.title("HLSP Hope Speech Detection Demo (Scaffold)")

text = st.text_area("Enter Roman Urdu text", height=120)
normalizer = PhoneticNormalizer()

if st.button("Analyze"):
    normalized = normalizer.normalize(text)
    st.subheader("Before Normalization")
    st.write(text)
    st.subheader("After Normalization")
    st.write(normalized)
    st.subheader("Predicted Class")
    st.info("Scaffold mode: model prediction not connected yet.")
