from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
loader = PyPDFLoader(r'C:\Users\GOURAV SHARMA\OneDrive\Documents\AI projects\RAG BASED PDF Q&A\paper.pdf')
pages = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size = 500 , chunk_overlap = 50,separators=['\n\n','\n',''])
chunks = splitter.split_documents(pages) 
embeddings = HuggingFaceEmbeddings(model_name = 'all-MiniLM-L6-v2')
vectorstore = Chroma.from_documents(chunks , embeddings)
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv('GROQ_API_KEY')
)
prompt = ChatPromptTemplate.from_template("""
Answer the question based on only the context below.
Always mention which part of the context you used.
Context : {context}
Question : {question}
""")
retriever = vectorstore.as_retriever(search_kwargs={'k':5})
def format_docs(docs):
    return '\n\n'.join([d.page_content for d in docs])
print("RAG is Ready! Type your questions now (Type quit to exit)")
while True:
    query = input('Your Questions : ')
    if query.lower() == 'quit' :
        break
    docs = retriever.invoke(query)
    Context = format_docs(docs)
    final_prompt = f"""
    Answer the question based only on the Context below.
    Always mention which part of the Context you used.
    Context : {Context}
    Question : {query}
    """
    result = llm.invoke(final_prompt)
    print("Answer : ")
    print(result.content)
    print("SOURCES : ")
    for i , doc in enumerate(docs):
        print(f"----Chunk {i+1} (Page {doc.metadata.get('page','?')})----")
        print(doc.page_content[:200])
    print('\n')