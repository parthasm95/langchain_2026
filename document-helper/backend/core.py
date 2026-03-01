# Retrieval Module
# Step-1: Initialize embedding model (must match ingestion embedding model + vector dimension)
# Step-2: Connect to Pinecone vector index using PineconeVectorStore
# Step-3: Convert user query into embedding vector
# Step-4: Perform similarity search (top-k most relevant chunks)
# Step-5: Serialize retrieved documents into readable context for LLM
# Step-6: Expose retrieval as a @tool so the agent can call it
# Step-7: Create agent with retrieval tool using create_agent()
# Step-8: Agent decides → calls retrieve_context tool → receives context
# Step-9: LLM generates grounded answer using retrieved context
# Step-10: Extract ToolMessage artifacts for debugging / citation display

import os
import sys
from typing import Any, Dict
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import ToolMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.chat_models import init_chat_model

load_dotenv()

# Initialize embedding model (must match ingestion embedding model + vector dimension)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# initialize the vector store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
)

# initialize chat model
chat_model = init_chat_model("gpt-4o-mini", model_provider="openai")

@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer the user's question about Langchain"""
    # Step 1: Retrieve top 4 most similar documents from the vector store
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # Step 2: Serialize the retrieved documents into a readable context for the LLM
    serialized = "\n\n".join(
        (f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}")
        for doc in retrieved_docs
    )

    # Step 3: Return the serialized context and the retrieved documents
    return serialized, retrieved_docs

    
def run_llm(query: str) -> Dict[str, Any]:
    """
    Run the RAG pipeline to answer the user's question from the retrieved context

    Args:
        query: The user's question

    Returns:
        A dictionary containing:
        - answer: The answer to the user's question
        - context: List of retrieved documents
    """
    # Step 1: Define System Prompt
    system_prompt = (
        "You are a helpful AI assistant that answers questions about LangChain documentation. "
        "You have access to a tool that retrieves relevant documentation. "
        "Use the tool to find relevant information before answering questions. "
        "Always cite the sources you use in your answers. "
        "If you cannot find the answer in the retrieved documentation, say so."
    )

    # Step 2: Create the agent with retrieval tool
    agent = create_agent(chat_model, tools=[retrieve_context], system_prompt=system_prompt)

    # Step 3: Build messages List
    messages = [{"role": "user", "content": query}]

    # Step 4: Invoke the agent
    response = agent.invoke({"messages": messages})

    # Step 5: Extract the answer from the last AI message
    answer = response.get("messages")[-1].content

    # Step 6: Extract context documents from the ToolMessage artifacts
    context_docs = []
    for message in response["messages"]:
        # Check if this is a ToolMessage with artifact
        if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
            # The artifact should contain the list of Document objects
            if isinstance(message.artifact, list):
                context_docs.extend(message.artifact)

    # Step 7: Return the answer and context documents
    return {
        "answer": answer,
        "context": context_docs,
    }

# Example usage
if __name__ == "__main__":
    result = run_llm(query="what are deep agents?")
    print(result)