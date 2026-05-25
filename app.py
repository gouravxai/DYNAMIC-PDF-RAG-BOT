import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
import tempfile
import os
import numpy as np
import io
from dotenv import load_dotenv
from pdf2image import convert_from_path
import base64

load_dotenv()

st.set_page_config(
    page_title="RAG PDF Q&A with Images",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("📄 RAG - PDF Q&A with Images & OCR")
st.markdown("*Upload PDFs with text or scanned pages. Ask questions and get answers with citations.*")

@st.cache_resource
def load_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.3
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_vision_llm():
    return ChatGroq(
        model="llama-3.2-11b-vision-preview",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.2
    )

def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        pdf_path = f.name

    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=400,
            chunk_overlap=100,
            separators=["\n\n", "\n", ".", " ", ""]
        )

        chunks = splitter.split_documents(pages)

        return chunks, pdf_path

    except Exception as e:
        st.error(f"Error processing PDF text: {str(e)}")
        return [], pdf_path

def extract_images_from_pdf(pdf_path, max_pages=None):
    try:
        images = convert_from_path(pdf_path, dpi=150)

        if max_pages:
            images = images[:max_pages]

        image_data = []
        progress_bar = st.progress(0)

        for idx, image in enumerate(images):
            img_bytes = io.BytesIO()
            image.save(img_bytes, format="PNG")
            img_bytes.seek(0)

            b64_string = base64.b64encode(img_bytes.getvalue()).decode()

            image_data.append({
                "page": idx + 1,
                "base64": b64_string,
                "size": len(b64_string)
            })

            progress_bar.progress((idx + 1) / len(images))

        return image_data

    except Exception as e:
        st.error(f"Could not extract images from PDF: {str(e)}")
        return []

def extract_text_from_images(image_data_list, vision_llm):
    extracted_texts = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, img_data in enumerate(image_data_list):
        try:
            status_text.text(
                f"📖 Processing page {img_data['page']}/{len(image_data_list)}..."
            )

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
                        "text": """Extract ALL text and information from this image.
Include all visible details, tables, specifications, labels, prices and codes."""
                    }
                ]
            )

            response = vision_llm.invoke([message])

            extracted_texts.append({
                "page": img_data["page"],
                "text": response.content,
                "source": "image_ocr"
            })

            progress_bar.progress((idx + 1) / len(image_data_list))

        except Exception as e:
            st.warning(f"⚠️ Error processing page {img_data['page']}: {str(e)}")

            extracted_texts.append({
                "page": img_data["page"],
                "text": f"[Error extracting text from page {img_data['page']}]",
                "source": "image_ocr"
            })

    status_text.empty()
    progress_bar.empty()

    return extracted_texts

def cosine_similarity(a, b):
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0

    return np.dot(a, b) / (norm_a * norm_b)

def retrieve(query, text_chunks, image_texts, embeddings_model, k=15):
    query_emb = np.array(embeddings_model.embed_query(query))

    all_chunks = []

    for chunk in text_chunks:
        all_chunks.append({
            "content": chunk.page_content,
            "page": chunk.metadata.get("page", "?"),
            "source": "text"
        })

    for img_text in image_texts:
        all_chunks.append({
            "content": img_text["text"],
            "page": img_text["page"],
            "source": "image_ocr"
        })

    if not all_chunks:
        return []

    chunk_texts = [c["content"] for c in all_chunks]

    try:
        chunk_embs = np.array(
            embeddings_model.embed_documents(chunk_texts)
        )

    except Exception as e:
        st.error(f"Error embedding documents: {str(e)}")
        return []

    scores = [
        cosine_similarity(query_emb, ce)
        for ce in chunk_embs
    ]

    top_k_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:k]

    return [
        (all_chunks[i], scores[i])
        for i in top_k_indices
    ]

if "messages" not in st.session_state:
    st.session_state.messages = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "image_texts" not in st.session_state:
    st.session_state.image_texts = []

if "current_file" not in st.session_state:
    st.session_state.current_file = None

llm = load_llm()
embeddings_model = load_embeddings()
vision_llm = load_vision_llm()

with st.sidebar:
    st.header("⚙️ Settings")

    retrieval_k = st.slider(
        "Number of sources to retrieve",
        min_value=5,
        max_value=20,
        value=10
    )

    show_scores = st.checkbox(
        "Show similarity scores",
        value=False
    )

    st.divider()

    if st.button("🔄 Clear Everything", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chunks = []
        st.session_state.image_texts = []
        st.session_state.current_file = None
        st.rerun()

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type="pdf"
)

if uploaded_file:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"

    if st.session_state.get("current_file") != file_id:
        st.session_state.current_file = file_id
        st.session_state.messages = []

        with st.spinner("📖 Processing PDF..."):
            file_bytes = uploaded_file.read()

            st.session_state.chunks, pdf_path = process_pdf(file_bytes)

        with st.spinner("🖼️ Extracting images..."):
            image_data = extract_images_from_pdf(pdf_path)

            if image_data:
                with st.spinner("🤖 Running OCR..."):
                    st.session_state.image_texts = (
                        extract_text_from_images(
                            image_data,
                            vision_llm
                        )
                    )

            else:
                st.session_state.image_texts = []

    st.success(f"✅ '{uploaded_file.name}' loaded!")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        with st.chat_message("user"):
            st.write(query)

        st.session_state.messages.append({
            "role": "user",
            "content": query
        })

        history = ""

        for msg in st.session_state.messages[-6:]:
            history += f"{msg['role']}: {msg['content']}\n"

        with st.spinner("🔍 Searching..."):
            docs_with_scores = retrieve(
                query,
                st.session_state.chunks,
                st.session_state.image_texts,
                embeddings_model,
                k=retrieval_k
            )

        if not docs_with_scores:
            st.warning("⚠️ No relevant content found")

        else:
            context = '\n\n'.join([
                d[0]["content"]
                for d in docs_with_scores
            ])

            final_prompt = f"""
You are a helpful assistant who answers questions about a PDF document.

Chat History:
{history}

PDF Content:
{context}

User Question:
{query}

Answer:
"""

            with st.spinner("🤔 Thinking..."):
                try:
                    result = llm.invoke(final_prompt)
                    answer = result.content

                except Exception as e:
                    answer = f"Error generating response: {str(e)}"

            with st.chat_message("assistant"):
                st.write(answer)

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer
            })

            with st.expander("📚 Sources Used"):
                for i, (doc, score) in enumerate(docs_with_scores, 1):
                    source_label = (
                        "📄 Text"
                        if doc["source"] == "text"
                        else "🖼️ Image"
                    )

                    st.markdown(
                        f"**{source_label} - Page {doc['page']}**"
                    )

                    if show_scores:
                        st.write(f"Score: {score:.3f}")

                    st.caption(
                        doc["content"][:400] + "..."
                        if len(doc["content"]) > 400
                        else doc["content"]
                    )

                    st.divider()

else:
    st.info("👆 Upload a PDF to get started!")
