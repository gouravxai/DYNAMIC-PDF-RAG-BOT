import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

st.title("RAG - PDF Q&A with Citations")

@st.cache_resource
def load_llm():
    return ChatGroq(model="llama-3.1-8b-instant", api_key=os.getenv("GROQ_API_KEY"))

def process_pdf(file_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(file_bytes)
        path = f.name
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore

llm = load_llm()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    if st.session_state.get("current_file") != uploaded_file.name:
        st.session_state.current_file = uploaded_file.name
        st.session_state.messages = []
        st.session_state.vectorstore = process_pdf(uploaded_file.read())

    st.success(f"{uploaded_file.name} uploaded!")
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

    def format_docs(docs):
        return '\n\n'.join([d.page_content for d in docs])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    query = st.chat_input("Ask a question about your PDF...")

    if query:
        with st.chat_message("user"):
            st.write(query)
        st.session_state.messages.append({"role": "user", "content": query})

        history = ""
        for msg in st.session_state.messages[-6:]:
            history += f"{msg['role']}: {msg['content']}\n"

        docs = retriever.invoke(query)
        context = format_docs(docs)

        final_prompt = f"""You are a helpful assistant. Follow these rules strictly:

1. Answer ONLY from the PDF context below. Never use chat history to form your answer.
2. Chat history is only to understand what the user is referring to in follow-up questions.
3. If the answer is not in the PDF, say "This isn't covered in the PDF."
4. Detect the language of the user's question and reply in the same language.
5. Keep answers natural and conversational. Do not use robotic phrases like "According to the context", "It is stated that", "Based on the provided text", "The context mentions". Just answer directly.
6. If the user is just chatting (hi, thanks, how are you etc), respond naturally without forcing PDF context.
7. At the end of your answer, add citations like: [Page 3], [Page 7]

Chat History (for follow-up reference only):
{history}

PDF Context:
{context}

Question: {query}
"""
        result = llm.invoke(final_prompt)
        answer = result.content

        with st.chat_message("assistant"):
            st.write(answer)
            with st.expander("Sources"):
                for i, doc in enumerate(docs):
                    st.markdown(f"**Chunk {i+1} - Page {doc.metadata.get('page', '?')}**")
                    st.caption(doc.page_content[:300])

        st.session_state.messages.append({"role": "assistant", "content": answer})
