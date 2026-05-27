import streamlit as st
import tempfile
import os
import io
import re
import base64
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from pdf2image import convert_from_path

load_dotenv()

st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")
st.title("📄 Advanced PDF RAG")

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

def process_pdf_content(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " "]
        )
        text_chunks = splitter.split_documents(pages)

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
    ocr_results = []
    bar = st.progress(0, text="Processing pages...")
    
    for i, img in enumerate(image_data_list):
        msg = HumanMessage(content=[
            {"type": "text", "text": "Extract all text, numbers, tables, and details from this page exactly as they appear."},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img['base64']}"}}
        ])
        try:
            res = vision_llm.invoke([msg])
            ocr_results.append({"page": img["page"], "text": res.content, "source": "OCR"})
        except Exception as e:
            st.warning(f"Vision processing failed on page {img['page']}: {e}")
        bar.progress((i + 1) / len(image_data_list))
    
    bar.empty()
    return ocr_results

def hybrid_retrieve(query, text_chunks, ocr_chunks, k=10):
    all_docs = []
    for c in text_chunks:
        all_docs.append({"content": c.page_content, "page": c.metadata.get("page", 0) + 1, "source": "Text"})
    all_docs.extend(ocr_chunks)

    codes = re.findall(r'\b\d{6}\b|\b[A-Z]{2,}[\dA-Z\-]{4,}\b', query.upper())
    
    scored_docs = []
    for doc in all_docs:
        content = doc["content"]
        content_upper = content.upper()
        score = 0
        
        for code in codes:
            if code in content_upper:
                score += 50.0
        
        query_tokens = re.findall(r'\w+', query.lower())
        score += sum(2.0 for t in query_tokens if t.lower() in content.lower())
        
        if query.lower() in content.lower():
            score += 10.0
        
        scored_docs.append((doc, score))

    scored_docs.sort(key=lambda x: x[1], reverse=True)
    results = [d for d in scored_docs if d[1] > 0][:k]
    
    return results

def verify_content_exists(search_term, context_text):
    search_upper = search_term.upper()
    context_upper = context_text.upper()
    return search_upper in context_upper

def extract_exact_info(search_term, context_text):
    prompt = f"""Extract ONLY the exact information related to '{search_term}' from the text below.

If '{search_term}' is NOT found, respond with: NOT_FOUND

If found, respond with exact details from the document:
Term: {search_term}
Details: [exact information from document]
Source: [exact line from document]

Document Text:
{context_text}"""
    
    response = llm.invoke(prompt)
    return response.content

def answer_query(query, context_text):
    prompt = f"""Answer the question using ONLY the document text provided below.
If the answer is not in the document, say "Information not found in document".
Be precise and only report what actually exists in the source.

Document:
{context_text}

Question: {query}"""
    
    response = llm.invoke(prompt)
    return response.content

if "messages" not in st.session_state:
    st.session_state.messages = []
if "pdf_data" not in st.session_state:
    st.session_state.pdf_data = None

with st.sidebar:
    st.header("Upload PDF")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file and st.button("Process PDF"):
        with st.spinner("Processing..."):
            text_chunks, image_data = process_pdf_content(uploaded_file)
            ocr_chunks = run_vision_ocr(image_data)
            st.session_state.pdf_data = {"text": text_chunks, "ocr": ocr_chunks}
            st.success("PDF indexed!")

if st.session_state.pdf_data:
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    if query := st.chat_input("Ask a question about the PDF..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        hits = hybrid_retrieve(
            query, 
            st.session_state.pdf_data["text"], 
            st.session_state.pdf_data["ocr"],
            k=10
        )
        
        if not hits:
            response_text = "❌ No relevant content found in the document."
        else:
            context = "\n---\n".join([d[0]['content'] for d in hits])
            
            codes = re.findall(r'\b\d{6}\b|\b[A-Z]{2,}[\dA-Z\-]{4,}\b', query.upper())
            
            if codes:
                search_term = codes[0]
                if verify_content_exists(search_term, context):
                    response_text = extract_exact_info(search_term, context)
                else:
                    response_text = f"❌ '{search_term}' not found in the document."
            else:
                response_text = answer_query(query, context)

        with st.chat_message("assistant"):
            st.markdown(response_text)
            with st.expander("📋 View Sources"):
                if hits:
                    for i, (doc, score) in enumerate(hits, 1):
                        st.write(f"**Source {i}** - Page {doc['page']} (Score: {score:.1f})")
                        st.code(doc['content'][:400] + "..." if len(doc['content']) > 400 else doc['content'])
                else:
                    st.write("No sources.")
        
        st.session_state.messages.append({"role": "assistant", "content": response_text})
else:
    st.info("Upload and process a PDF in the sidebar to start.")
