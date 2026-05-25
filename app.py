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

# ============================================================================
# CACHE MODELS
# ============================================================================

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

# ============================================================================
# PDF PROCESSING
# ============================================================================

def process_pdf(file_bytes):
    """Extract text chunks from PDF"""
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as f:
        f.write(file_bytes)
        pdf_path = f.name

    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    # Smaller chunks for catalog data
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # FIXED: Was 1500, too large
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    chunks = splitter.split_documents(pages)

    return chunks, pdf_path

def extract_images_from_pdf(pdf_path):
    """Convert PDF pages to images"""
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
        st.error("Install poppler: sudo apt-get install poppler-utils")
        return []

def extract_text_from_images(image_data_list):
    """Run OCR on images"""
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
                        "text": """Extract ALL visible text exactly as shown.

CRITICAL:
- Preserve SKU codes EXACTLY
- Preserve product codes EXACTLY
- Preserve prices EXACTLY
- Preserve ratings EXACTLY
- Preserve tables EXACTLY
- Do NOT summarize or interpret
- Do NOT change formatting
- Include ALL text you can read"""
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
            st.warning(f"Error processing page {img_data['page']}: {str(e)}")
            extracted_texts.append({
                "page": img_data["page"],
                "text": "",
                "source": "image_ocr"
            })

    progress.empty()
    return extracted_texts

# ============================================================================
# RETRIEVAL & SEARCH
# ============================================================================

def cosine_similarity(a, b):
    """Compute cosine similarity"""
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
    """
    Hybrid retrieval: exact match + semantic search
    
    FIXED: Better scoring logic for catalog data
    """
    query_lower = query.lower()
    all_chunks = []

    # Add text chunks
    for chunk in text_chunks:
        all_chunks.append({
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", "?"),
            "source": "text"
        })

    # Add OCR chunks
    for item in image_texts:
        all_chunks.append({
            "content": item["text"],
            "page": item["page"],
            "source": "image_ocr"
        })

    if not all_chunks:
        return []

    # ========================================================================
    # IMPROVED EXACT MATCH SCORING
    # ========================================================================
    
    query_tokens = re.findall(r'\w+', query_lower)
    exact_matches = []

    for chunk in all_chunks:
        content_lower = chunk["content"].lower()
        
        # FIXED: Better scoring logic
        score = 0
        
        # Highest priority: full query appears as substring
        if query_lower in content_lower:
            score = 10000
        # High priority: all tokens present
        elif all(token in content_lower for token in query_tokens):
            score = 1000
        # Medium priority: count token occurrences
        else:
            score = sum(
                content_lower.count(token) * len(token)
                for token in query_tokens
            )

        if score > 0:
            exact_matches.append((chunk, score))

    # If we have enough exact matches, return them
    if len(exact_matches) >= k:
        exact_matches = sorted(
            exact_matches,
            key=lambda x: x[1],
            reverse=True
        )
        return exact_matches[:k]

    # ========================================================================
    # SEMANTIC SEARCH (if not enough exact matches)
    # ========================================================================
    
    query_embedding = np.array(
        embeddings_model.embed_query(query)
    )

    chunk_texts = [c["content"] for c in all_chunks]
    chunk_embeddings = np.array(
        embeddings_model.embed_documents(chunk_texts)
    )

    semantic_scores = [
        cosine_similarity(query_embedding, emb)
        for emb in chunk_embeddings
    ]

    semantic_results = [
        (all_chunks[i], semantic_scores[i] * 1000)  # Scale to be comparable
        for i in range(len(all_chunks))
    ]

    # ========================================================================
    # COMBINE & DEDUPLICATE
    # ========================================================================
    
    combined = exact_matches + semantic_results
    
    final = []
    seen = set()

    for item in sorted(
        combined,
        key=lambda x: x[1],
        reverse=True
    ):
        # FIXED: Better deduplication logic
        key = (item[0]["page"], item[0]["source"])
        
        if key not in seen:
            seen.add(key)
            final.append(item)

    return final[:k]

# ============================================================================
# SESSION STATE & UI
# ============================================================================

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "image_texts" not in st.session_state:
    st.session_state.image_texts = []

if "current_file" not in st.session_state:  # FIXED: Track current file
    st.session_state.current_file = None

# ============================================================================
# FILE UPLOAD & PROCESSING
# ============================================================================

uploaded_file = st.file_uploader(
    "📄 Upload PDF Catalog",
    type="pdf"
)

if uploaded_file:
    # FIXED: Only process if NEW file
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.get("current_file") != file_id:
        st.session_state.current_file = file_id
        st.session_state.messages = []  # Clear chat for new PDF
        
        # Process PDF
        with st.spinner("📖 Processing PDF..."):
            file_bytes = uploaded_file.read()
            chunks, pdf_path = process_pdf(file_bytes)
            st.session_state.chunks = chunks

        # Extract images
        with st.spinner("🖼️ Extracting images..."):
            image_data = extract_images_from_pdf(pdf_path)

        # Run OCR
        if image_data:
            with st.spinner("🤖 Running OCR..."):
                image_texts = extract_text_from_images(image_data)
                st.session_state.image_texts = image_texts
        else:
            st.session_state.image_texts = []

        st.success(f"✅ PDF loaded! ({len(chunks)} text chunks + {len(image_data)} pages)")

    # ========================================================================
    # CHAT INTERFACE
    # ========================================================================

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input
    query = st.chat_input("🔍 Ask about SKU, product, price, rating...")

    if query:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        with st.chat_message("user"):
            st.write(query)

        # Retrieve relevant chunks
        with st.spinner("🔍 Searching catalog..."):
            results = retrieve(
                query,
                st.session_state.chunks,
                st.session_state.image_texts,
                embeddings_model,
                k=12
            )

        if not results:
            st.warning("⚠️ No relevant catalog data found")
        else:
            # Build context
            context = "\n\n".join([
                r[0]["content"]
                for r in results
            ])

            # FIXED: Strict prompt for catalog lookup
            prompt = f"""You are an EXACT catalog lookup assistant.

CRITICAL INSTRUCTIONS:
1. Return ONLY information found in the catalog
2. Do NOT guess, assume, or interpolate
3. Do NOT invent SKU codes or prices
4. If exact match not found, respond:
   "Exact match not found in catalog"
5. If price/rating/spec is missing, say so explicitly
6. Always cite the page number
7. Format clearly with fields:
   - SKU Code
   - Product Name
   - Specifications
   - Price
   - Rating
   - Page Number
8. If multiple products match, list all with clear separation

CATALOG DATA PROVIDED:
{context}

USER QUERY: {query}

RESPONSE (be exact and cite sources):
"""

            # Generate answer
            with st.spinner("✍️ Generating answer..."):
                response = llm.invoke(prompt)
                answer = response.content

            # Display answer
            with st.chat_message("assistant"):
                st.write(answer)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            # Show sources
            with st.expander("📚 Retrieved Sources"):
                for idx, (doc, score) in enumerate(results, 1):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**Page {doc['page']} | {doc['source'].upper()}**")
                    with col2:
                        st.metric("Score", f"{score:.0f}")

                    with st.container(border=True):
                        st.code(doc["content"][:800])

else:
    st.info("👆 Upload a PDF to begin searching the catalog")
