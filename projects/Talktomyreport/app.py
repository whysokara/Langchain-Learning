import streamlit as st
from pdf_parser import extract_text_by_page

st.title("🩺 Parse Your Medical Report")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pages = extract_text_by_page(uploaded_file)
    st.success("PDF parsed successfully!")
    
    # Preview first page
    st.subheader("Preview - Page 1")
    st.write(pages[0])

    # Optional: Total pages info
    st.caption(f"Total pages extracted: {len(pages)}")
