import streamlit as st
import tempfile
import os
import io
import re
import base64
import numpy as np
from dotenv import load_dotenv

# PDF & LangChain
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path

load_dotenv()

# --- Page Config ---
st.set_page_config(page_title="Advanced PDF RAG", page_icon="📄", layout="wide")
st.title("📄 Advanced PDF Catalog RAG")

# --- Model Loading (Cached) ---
@st.cache_resource
def load_models():
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    vision_llm = ChatGroq(model="llama-3.2-11b-vision-preview", temperature=0)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return llm, vision_llm, embeddings

try:
    llm, vision_llm, embeddings_model = load_models()
except Exception as e:
    st.error(f"Error connecting to Groq: {e}. Check your API Key.")

# --- Processing Logic ---
def process_pdf_content(uploaded_file):
    """Saves uploaded file to temp path and processes text/images"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # 1. Extract Text Chunks
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        text_chunks = splitter.split_documents(pages)

        # 2. Extract Images for OCR
        images = convert_from_path(tmp_path, dpi=150)
        image_data = []
        for i, img in enumerate(images):
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            image_data.append({"page": i + 1, "base64": b64})

        return text_chunks, image_data
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

def run_vision_ocr(image_data_list):
    """Converts images to text using Vision LLM"""
    ocr_results = []
    bar = st.progress(0, text="Analyzing catalog pages...")
    
    for i, img in enumerate(image_data_list):
        msg = HumanMessage(content=[
            {"type": "text", "text": "Extract all SKU codes, prices, and product details exactly. Use a table format if possible."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}}
        ])
        try:
            res = vision_llm.invoke([msg])
            ocr_results.append({"page": img["page"], "text": res.content, "source": "OCR"})
        except Exception as e:
            st.warning(f"Vision failed on page {img['page']}: {e}")
        bar.progress((i + 1) / len(image_data_list))
    
    bar.empty()
    return ocr_results

def hybrid_retrieve(query, text_chunks, ocr_chunks, k=8):
    """Simple but effective hybrid retrieval"""
    all_docs = []
    for c in text_chunks:
        all_docs.append({"content": c.page_content, "page": c.metadata.get("page", 0) + 1, "source": "Text"})
    all_docs.extend(ocr_chunks)

    # 1. Keyword Scoring
    query_tokens = re.findall(r'\w+', query.lower())
    scored_docs = []
    for doc in all_docs:
        content = doc["content"].lower()
        score = sum(2.0 for t in query_tokens if t in content) # Basic keyword match
        if query.lower() in content: score += 10.0 # Phrase match
        scored_docs.append((doc, score))

    # 2. Sort and Deduplicate
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    # Take top K results that actually have a score > 0
    results = [d for d in scored_docs if d[1] > 0][:k]
    
    # If no keyword matches, just return top snippets (fallback)
    if not results:
        results = scored_docs[:3]
        
    return results

# --- UI Layout ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "catalog_data" not in st.session_state:
    st.session_state.catalog_data = None

with st.sidebar:
    st.header("Upload Center")
    uploaded_file = st.file_uploader("Upload Catalog PDF", type="pdf")
    if uploaded_file and st.button("Process Catalog"):
        with st.spinner("Processing..."):
            text_chunks, image_data = process_pdf_content(uploaded_file)
            ocr_chunks = run_vision_ocr(image_data)
            st.session_state.catalog_data = {"text": text_chunks, "ocr": ocr_chunks}
            st.success("Catalog indexed!")

# --- Chat Interface ---
if st.session_state.catalog_data:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("What SKU are you looking for?"):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # Retrieval
        hits = hybrid_retrieve(
            query, 
            st.session_state.catalog_data["text"], 
            st.session_state.catalog_data["ocr"]
        )
        
        context = "\n---\n".join([f"Source: Page {d[0]['page']}\n{d[0]['content']}" for d in hits])
        
        prompt = f"""Use the following catalog excerpts to answer. 
        If the SKU or product is not listed, say 'Not found in catalog'.
        Format as a clean list or table.
        
        Context:
        {context}
        
        User Query: {query}"""

        with st.chat_message("assistant"):
            response = llm.invoke(prompt)
            st.markdown(response.content)
            with st.expander("View Sources"):
                for doc, score in hits:
                    st.write(f"**Page {doc['page']}** (Score: {score})")
                    st.code(doc['content'][:300] + "...")
        
        st.session_state.messages.append({"role": "assistant", "content": response.content})
else:
    st.info("Please upload and process a PDF catalog in the sidebar to start.")
