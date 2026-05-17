# rag-pdf-qa

A RAG (Retrieval-Augmented Generation) system that lets you upload any PDF and ask questions about it. It extracts answers strictly from the document context and provides the exact page citations for transparency.

Live Demo: [Dynamic PDF RAG Bot](https://dynamic-pdf-rag-bot-xsbozwiqebofdkwf8txbwk.streamlit.app/)

## Features
* **Dynamic File Upload:** Users can upload any custom PDF document directly through the interface for real-time indexing.
* **Contextual Chat Memory:** Maintains conversation history within a session to support multi-turn, follow-up questions.
* **Strict Context Boundaries:** The LLM is restricted to answering solely based on the retrieved document segments to eliminate hallucinations.
* **Source Citations:** Displays the specific document chunks and exact PDF page numbers used to formulate the response.

## Architecture & Logic
* **Document Ingestion:** The uploaded PDF is parsed and broken into semantic units using a `RecursiveCharacterTextSplitter` with a chunk size of 500 tokens and a 50-token overlap.
* **Vector Store:** Text chunks are converted into vector embeddings using the `all-MiniLM-L6-v2` model and indexed into an in-memory `ChromaDB` store.
* **Retrieval & Inference:** When a user submits a query, a cosine similarity search retrieves the top 5 most relevant chunks. These chunks, along with the conversation history and the query, are compiled into a custom prompt template and processed via `Llama-3.1-8b-instant` on Groq Cloud.

## Technical Stack
* **Language:** Python
* **LLM Orchestration:** LangChain
* **Vector Database:** ChromaDB
* **Embeddings Model:** HuggingFace (`all-MiniLM-L6-v2`)
* **Inference Engine:** ChatGroq (`llama-3.1-8b-instant`)
* **Interface:** Streamlit

## Challenges Faced
* **Windows Path Escapes:** Encountered system errors during local testing due to Python treating `\U` in file paths as Unicode escape characters. Resolved by enforcing raw string literals (`r'path'`).
* **LangChain Abstraction Deprecations:** Standard high-level components like `RetrievalQA` proved unstable across LangChain version updates. Solved this by decoupling the pipeline and manually handling the retrieval, formatting, and invocation steps.
* **API Quota Constraints:** The initial integration with Google's Gemini free tier hit strict rate limits during iterative testing. Migrated the backend to Groq Cloud to leverage faster inference and higher token-per-minute thresholds.
* **Context Window Optimization:** Initial trials with 3 retrieved chunks occasionally cut off multi-page tables or continuous context in dense papers. Fine-tuned the retriever parameter to `k=5` and optimized chunk spacing to prevent data loss.

## Local Setup
1. Install dependencies:
   ```bash
   pip install streamlit langchain langchain-community langchain-huggingface langchain-groq chromadb pypdf python-dotenv sentence-transformers langchain-text-splitters
