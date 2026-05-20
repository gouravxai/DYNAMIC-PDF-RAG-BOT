## Dynamic PDF RAG Bot — Context-Aware Document Intelligence

Live Demo: [Try it here](https://dynamic-pdf-rag-bot-xsbozwiqebofdkwf8txbwk.streamlit.app/)

### Overview

I built a Retrieval-Augmented Generation system that lets you upload any PDF and have a grounded conversation with its content. Unlike standard LLMs that hallucinate, this bot is forced to stay within the document boundaries and provides page numbers for every answer.

The key difference: it doesn't make things up. Every answer is backed by actual text from your document.

### The Problem

Large language models are great at generating text, but they have a critical flaw - they often confabulate facts when they don't know something. This makes them risky for analyzing important documents like research papers, legal contracts, or technical manuals where accuracy matters.

I wanted to build something that couldn't hallucinate because it has nowhere to hallucinate from.

### How It Works

The system has three main parts:

**1. Document Processing**
- PDFs are split into 500-token chunks with 50-token overlap to preserve context
- This overlap is important because it keeps related information from being separated

**2. Vector Search**
- Uses HuggingFace embeddings (all-MiniLM-L6-v2) to convert chunks into vectors
- Stores them in ChromaDB for fast similarity search
- When you ask a question, it finds the 5 most relevant chunks using cosine similarity

**3. Answer Generation**
- Takes those relevant chunks and feeds them to Llama-3.1-8b via Groq
- The model generates an answer based only on what's in those chunks
- Returns page numbers and quotes so you can verify everything

### Technical Stack

- LLM Orchestration: LangChain
- LLM Backend: Llama-3.1-8b (via Groq Cloud)
- Vector Database: ChromaDB
- Embeddings: HuggingFace (all-MiniLM-L6-v2)
- Frontend: Streamlit
- API: Groq Cloud

### Challenges I Solved

**Challenge 1: API Rate Limiting**
Initially used Google's speech APIs but hit rate limits quickly. Switched to Groq which gives 10x faster inference and higher rate limits. This was critical for handling multiple users.

**Challenge 2: Context Loss in Retrieval**
At first, I set the retriever to fetch k=3 chunks, which wasn't enough for dense documents with tables. Increasing to k=5 fixed it without significantly slowing down the system.

**Challenge 3: LangChain Abstractions Breaking**
The RetrievalQA abstraction kept causing issues with prompt formatting. I decoupled it and handled retrieval and formatting manually, giving me more control and reliability.

### Performance

- Transcription to answer: typically 2-3 seconds
- Vector search: sub-100ms
- Handles large PDFs (100+ pages) without issues
- Runs on free Streamlit Cloud tier

### Installation & Usage

```bash
git clone https://github.com/gouravxai/DYNAMIC-PDF-RAG-BOT.git
cd DYNAMIC-PDF-RAG-BOT

pip install -r requirements.txt

# Create .env file
echo "GROQ_API_KEY=your_key_here" > .env

streamlit run app.py
```

Then visit the live demo above to test it without installation.

### What I Learned

The biggest insight was understanding the tradeoff between retrieval depth and speed. Fetching more context chunks helps accuracy but adds latency. With RAG systems, this balance matters more than raw model size.

Also learned that building for real users means handling edge cases - PDFs with weird formatting, tables, multiple columns. The system needed to be robust, not just accurate on clean data.

### Future Improvements

- Support for multi-language PDFs
- Ability to upload multiple PDFs and search across them
- Export conversation as a PDF report
- Fine-tuning embeddings on domain-specific data for better retrieval
