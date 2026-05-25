import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

import tempfile
import os
import io
import re
import base64
import numpy as np

from dotenv import load_dotenv
from pdf2image import convert_from_path

load_dotenv()

st.set_page_config(
    page_title="Advanced PDF RAG",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Advanced PDF Catalog RAG")

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

@st.cache_resource
def load_vision_llm():
    return ChatGroq(
        model="llama-3.2-11b-vision-preview",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

llm = load_llm()
vision_llm = load_vision_llm()
embeddings_model = load_embeddings()

def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as f:
        f.write(file_bytes)
        pdf_path = f.name

    loader = PyPDFLoader(pdf_path)

    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n"]
    )

    chunks = splitter.split_documents(pages)

    return chunks, pdf_path

def extract_images_from_pdf(pdf_path):
    try:
        images = convert_from_path(
            pdf_path,
            dpi=150
        )

        image_data = []

        for idx, image in enumerate(images):
            img_bytes = io.BytesIO()

            image.save(img_bytes, format="PNG")

            img_bytes.seek(0)

            b64_string = base64.b64encode(
                img_bytes.getvalue()
            ).decode()

            image_data.append({
                "page": idx + 1,
                "base64": b64_string
            })

        return image_data

    except Exception as e:
        st.error(f"PDF Image Extraction Error: {e}")
        st.error("Poppler is missing.")
        return []

def extract_text_from_images(image_data_list):
    extracted_texts = []

    progress = st.progress(0)

    for idx, img_data in enumerate(image_data_list):
        try:
            message = HumanMessage(
                content=[
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_data['base64']}"
                        }
                    },
                    {
                        "type": "text",
                        "text": """
Extract ALL visible text exactly.

IMPORTANT:
- Preserve SKU codes exactly
- Preserve tables exactly
- Preserve product codes exactly
- Preserve ratings, prices, models
- Do not summarize
"""
                    }
                ]
            )

            response = vision_llm.invoke([message])

            extracted_texts.append({
                "page": img_data["page"],
                "text": response.content,
                "source": "image_ocr"
            })

            progress.progress(
                (idx + 1) / len(image_data_list)
            )

        except Exception as e:
            extracted_texts.append({
                "page": img_data["page"],
                "text": "",
                "source": "image_ocr"
            })

    progress.empty()

    return extracted_texts

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return np.dot(a, b) / (norm_a * norm_b)

def retrieve(
    query,
    text_chunks,
    image_texts,
    embeddings_model,
    k=12
):
    query_lower = query.lower()

    all_chunks = []

    for chunk in text_chunks:
        all_chunks.append({
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", "?"),
            "source": "text"
        })

    for item in image_texts:
        all_chunks.append({
            "content": item["text"],
            "page": item["page"],
            "source": "image_ocr"
        })

    if not all_chunks:
        return []

    query_tokens = re.findall(r'\w+', query_lower)

    exact_matches = []

    for chunk in all_chunks:
        content_lower = chunk["content"].lower()

        score = 0

        for token in query_tokens:
            if token in content_lower:
                score += 1

        if score > 0:
            exact_matches.append(
                (chunk, score * 100)
            )

    if len(exact_matches) >= k:
        exact_matches = sorted(
            exact_matches,
            key=lambda x: x[1],
            reverse=True
        )

        return exact_matches[:k]

    query_embedding = np.array(
        embeddings_model.embed_query(query)
    )

    chunk_texts = [
        c["content"]
        for c in all_chunks
    ]

    chunk_embeddings = np.array(
        embeddings_model.embed_documents(
            chunk_texts
        )
    )

    semantic_scores = [
        cosine_similarity(
            query_embedding,
            emb
        )
        for emb in chunk_embeddings
    ]

    semantic_results = [
        (all_chunks[i], semantic_scores[i])
        for i in range(len(all_chunks))
    ]

    combined = exact_matches + semantic_results

    final = []

    seen = set()

    for item in sorted(
        combined,
        key=lambda x: x[1],
        reverse=True
    ):
        key = item[0]["content"][:300]

        if key not in seen:
            seen.add(key)
            final.append(item)

    return final[:k]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "image_texts" not in st.session_state:
    st.session_state.image_texts = []

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file:
    with st.spinner("Processing PDF..."):
        file_bytes = uploaded_file.read()

        chunks, pdf_path = process_pdf(
            file_bytes
        )

        st.session_state.chunks = chunks

    with st.spinner("Extracting images..."):
        image_data = extract_images_from_pdf(
            pdf_path
        )

    with st.spinner("Running OCR..."):
        image_texts = extract_text_from_images(
            image_data
        )

        st.session_state.image_texts = image_texts

    st.success("PDF Loaded Successfully")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input(
        "Ask about SKU, MCCB, products..."
    )

    if query:
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        with st.spinner("Searching catalog..."):
            results = retrieve(
                query,
                st.session_state.chunks,
                st.session_state.image_texts,
                embeddings_model
            )

        context = "\n\n".join([
            r[0]["content"]
            for r in results
        ])

        prompt = f"""
You are a professional electrical product catalog assistant.

STRICT RULES:
1. NEVER invent SKU codes
2. NEVER guess nearby products
3. ONLY answer from provided context
4. If exact match not found say:
   "Exact match not found in PDF."
5. SKU codes must match EXACTLY
6. Product names must match EXACTLY
7. Always provide:
   - SKU
   - Product name
   - Ratings
   - Price
   - Page number
8. Format answers cleanly
9. If multiple matches exist, list all

CATALOG DATA:
{context}

USER QUERY:
{query}

ANSWER:
"""

        with st.spinner("Generating answer..."):
            response = llm.invoke(prompt)

            answer = response.content

        with st.chat_message("assistant"):
            st.write(answer)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

        with st.expander("Retrieved Sources"):
            for idx, (doc, score) in enumerate(results):
                st.markdown(
                    f"### Page {doc['page']} | Score: {score}"
                )

                st.code(doc["content"][:1500])

else:
    st.info("Upload a PDF to begin")
