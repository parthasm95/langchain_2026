import os
import pinecone
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

if __name__ == "__main__":
    print("Hello from Ingestion!")
    loader = TextLoader("/Users/partha/Cursor/langchain_2026/RAG-GIST/mediumblog1.txt")
    documents = loader.load()

    print("Splitting documents into chunks...")
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    print("Embedding chunks...")
    embeddings = OpenAIEmbeddings()

    print("Initializing Pinecone vector store...")
    vectorstore = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"), embedding=embeddings
    )

    print("Adding chunks to Pinecone vector store...")
    vectorstore.add_documents(chunks)
    print("Documents ingested successfully!")
