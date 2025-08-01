import fitz

def extract_text_by_page(uploaded_file):

    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    pages = [page.get_text() for page in doc]
    return pages