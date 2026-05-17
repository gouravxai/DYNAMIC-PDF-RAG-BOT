PROJECT: DYNAMIC PDF RAG BOT (CONTEXT-AWARE DOCUMENT INTELLIGENCE)


LIVE DEMO: 
https://dynamic-pdf-rag-bot-xsbozwiqebofdkwf8txbwk.streamlit.app/

OVERVIEW:
I built a Retrieval-Augmented Generation (RAG) system that allows users to 
upload any PDF and have a grounded conversation with its content. Unlike 
standard LLMs, this bot strictly cites its sources and provides page numbers, 
ensuring 100% transparency and zero hallucinations.

THE PROBLEM:
General-purpose LLMs often "hallucinate" facts when they don't know the answer. 
I wanted to build a system where the AI is forced to stay within the boundaries 
of a specific document, making it useful for analyzing research papers, legal 
contracts, or technical manuals.

ARCHITECTURE & LOGIC:
1. Ingestion Pipeline: Used RecursiveCharacterTextSplitter to break PDFs into 
   500-token chunks with a 50-token overlap to maintain semantic continuity.
2. Vector Store & Retrieval:
   - Embeddings: HuggingFace's all-MiniLM-L6-v2 for high-dimensional vectors.
   - Storage: Chunks indexed in ChromaDB for fast similarity searches.
   - Logic: Cosine similarity search to fetch the top 5 relevant segments.
3. Inference: Integrated Llama-3.1-8b via Groq Cloud for near-instant 
   response generation using a custom LangChain prompt template.

CHALLENGES OVERCOME:
- API Rate Limiting: Migrated from Gemini to Groq Cloud to leverage faster 
  inference and higher token-per-minute thresholds.
- Context Loss: Fine-tuned retriever to k=5 to ensure dense documents and 
  complex tables are fully captured without data loss.
- System Stability: Decoupled the pipeline from unstable LangChain 
  abstractions (like RetrievalQA) to handle retrieval and formatting manually.

TECHNICAL STACK:
- Orchestration: LangChain
- LLM: Llama-3.1-8b (via Groq)
- Vector DB: ChromaDB
- Embeddings: HuggingFace (all-MiniLM-L6-v2)
- Interface: Streamlit

