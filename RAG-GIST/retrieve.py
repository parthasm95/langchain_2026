import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage # to pass the query to the llm
from langchain_core.runnables import RunnablePassthrough # to pass the context and question to the llm
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # to initialize the llm and embeddings
from langchain_pinecone import PineconeVectorStore # to initialize the vector store
from langchain_core.prompts import ChatPromptTemplate # to initialize the prompt template
from langchain_core.output_parsers import StrOutputParser # to parse the output of the llm

load_dotenv()

# initialize the embeddings
embeddings = OpenAIEmbeddings()
vectorstore = PineconeVectorStore(index_name=os.getenv("INDEX_NAME"), embedding=embeddings)

# initialize the llm
llm = ChatOpenAI(model="gpt-4o-mini")

# retriever is the vector store as a retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3}) 

# initialize the prompt template
prompt_template = ChatPromptTemplate.from_template(
    """You are a helpful assistant that can answer questions about the following context only:
    {context}
    Question: {question}
    Provide a detailed answer to the question.
    """
)

# Function to format the documents into a string
def format_docs(docs):
    """Format the documents into a string"""
    return "\n\n".join(doc.page_content for doc in docs)

# ========================================================
# Option 2: Function Based Approach without LCEL
# ========================================================
def retrieval_chain_without_lcel(query: str):
    """
    Simple retrieval chain without LCEL.
    Manually retrieve the documents, format them and generate the answer.
    """
    # Step 1: Retrieve the documents
    docs = retriever.invoke(query)

    # Step 2: Format the documents
    context = format_docs(docs)

    # Step 3: Format the prompt template to get the messages, retuens a list of messages
    messages = prompt_template.format_messages(context=context, question=query)
    
    # step 4: invoke the llm with formatted messages
    result = llm.invoke(messages)

    # step 5: return the result
    return result.content

# ========================================================
# Option 3: Implementation with LCEL
# ========================================================
def retrieval_chain_with_lcel(query: str):
    """
    Retrieval chain with LCEL.
    Use the retriever, prompt template and llm to generate the answer.
    """
    retrival_chain = retriever | format_docs | prompt_template | llm | StrOutputParser()
    return retrival_chain

if __name__ == "__main__":

    # query
    query = "What is Pinecone in machine learning in short?"

    # ========================================================
    # Option 1: Raw invocation without RAG
    # ========================================================

    print("\n" + "="*50)
    print("Raw invocation without RAG")
    print("="*50)
    result = llm.invoke([HumanMessage(content=query)])
    print("\nAnswer:")
    print(result.content)

    # ========================================================
    # Option 2: Implementation without LCEL (simple Function Based approach)
    # ========================================================

    print("\n" + "="*50)
    print("RAG pipeline without LCEL")
    print("="*50)
    result_without_lcel = retrieval_chain_without_lcel(query)
    print("\nAnswer:")
    print(result_without_lcel)