Dynamic PDF RAG Bot: Context-Aware Document Intelligence
I built a Retrieval-Augmented Generation (RAG) system that allows users to upload any PDF and have a grounded conversation with its content. Unlike standard LLMs, this bot strictly cites its sources and provides page numbers, ensuring 100% transparency and zero hallucinations.

The Problem
General-purpose LLMs often "hallucinate" facts when they don't know the answer. I wanted to build a system where the AI is forced to stay within the boundaries of a specific document, making it useful for analyzing research papers, legal contracts, or technical manuals.

How I Built It (The Architecture)
Ingestion Pipeline: Used RecursiveCharacterTextSplitter to break PDFs into 500-token chunks with a 50-token overlap to maintain semantic continuity between pages.

Vector Store & Retrieval:

Embeddings: Used HuggingFace's all-MiniLM-L6-v2 to convert text into high-dimensional vectors.

Storage: Chunks are indexed in ChromaDB for fast similarity searches.

Logic: When a query is asked, the system performs a cosine similarity search to fetch the top 5 most relevant segments.

Inference: Integrated Llama-3.1-8b via Groq Cloud for near-instant response generation, using a custom LangChain prompt template that includes chat history for multi-turn conversations.

Challenges I Overcame
API Rate Limiting: Initially used Gemini, but hit strict quotas during testing. Migrated the backend to Groq Cloud, which significantly improved inference speed and token limits.

Context Loss: Noticed that retrieving only 3 chunks often missed data from complex tables. Fine-tuned the retriever to k=5 to ensure dense documents are fully captured.

System Stability: Moved away from high-level LangChain abstractions (like RetrievalQA) that were prone to deprecation errors. Instead, I manually decoupled the retrieval and formatting steps for a more stable, production-ready pipeline.

Tech Stack
Orchestration: LangChain

LLM: Llama-3.1-8b (Groq)

Vector DB: ChromaDB

Embeddings: HuggingFace

Interface: Streamlit
