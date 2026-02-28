# Ingestion module
# Step-1:Using tavily crawl to ingest the documentation
# Step-2: Split the text into chunks with RecursiveCharacterTextSplitter
# Step-3: Embed the chunks into a vector space with OpenAIEmbeddings
# Step-4: Store the chunks in a Pineconevector database with PineconeVectorStore


import asyncio
import os
import ssl
import traceback
from typing import List, Any, Dict

import certifi
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl


from logger import Colors, log_info, log_success, log_error, log_warning, log_header

load_dotenv()

# initialize the embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10
)

# initialize the vector store
vectorstore = PineconeVectorStore(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings,
)

# initialize the tavily crawl
tavily_crawl = TavilyCrawl(
    api_key=os.getenv("TAVILY_API_KEY"),
    show_progress_bar=False,
    retry_min_seconds=10
)

# async function to index documents from tavily crawl
async def index_documents(documents: List[Document], batch_size: int = 50):
    """Process documents in batches and index them into the vector store"""
    log_header("VECTOR INDEXING PROCESS")
    log_info(f"Vectorstore Indexing: Preparing to add {len(documents)} documents to the vectorstore", Colors.DARKCYAN)

    # create batches of documents
    batches = [
        documents[i:i+batch_size] for i in range(0, len(documents), batch_size)
    ]
    log_info(f"Vectorstore Indexing: Created {len(batches)} batches of {batch_size} documents", Colors.DARKCYAN)

    # process all batches concurrently
    async def add_batch_to_vectorstore(batch: List[Document], batch_index: int):
        """Add a batch of documents to the vector store (batch_index is 0-based)."""
        one_based = batch_index + 1
        try:
            log_info(f"Vectorstore Indexing: Adding batch {one_based} of {len(batches)} to the vectorstore", Colors.DARKCYAN)
            await vectorstore.aadd_documents(batch)
            log_success(f"Vectorstore Indexing: Batch {one_based} of {len(batches)} added to the vectorstore")
        except Exception as e:
            log_error(f"Vectorstore Indexing: Error adding batch {one_based} of {len(batches)} to the vectorstore: {e}")
            log_error(traceback.format_exc())
            return False
        return True

    # process batches concurrently (pass 0-based index so logs show "batch 1 of N")
    tasks = [add_batch_to_vectorstore(batch, i) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # log any exceptions returned by gather (raised inside tasks, not caught)
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            one_based = i + 1
            log_error(f"Vectorstore Indexing: Batch {one_based} failed with: {result}")
            log_error("".join(traceback.format_exception(type(result), result, result.__traceback__)))

    # count the number of successful batches
    successful_batches = sum(1 for result in results if result is True)

    if successful_batches == len(batches):
        log_success(
            f"VectorStore Indexing: All batches processed successfully! ({successful_batches}/{len(batches)})"
        )
    else:
        log_warning(
            f"VectorStore Indexing: Processed {successful_batches}/{len(batches)} batches successfully"
        )
    return successful_batches == len(batches)


async def main():        
    """Main function to orchestrate the complete ingestion process"""
    log_header("DOCUMENT INGESTION PIPELINE")

    log_info("Document Ingestion: Starting to ingest documents", Colors.PURPLE)

    tavily_crawl_results = tavily_crawl.invoke(
        input={
            "url": "https://python.langchain.com/",
            "extract_depth": "advanced",
            "instructions": "Documentatin relevant to ai agents",
            "max_depth": 3,
        }
    )

    if tavily_crawl_results.get("error"):
        log_error(f"TavilyCrawl: Error crawling documentation site: {tavily_crawl_results['error']}")
        return
    else:
        log_success(
            f"TavilyCrawl: Successfully crawled {len(tavily_crawl_results)} URLs from documentation site"
        )
    
    all_documents = []
    for tavily_crawl_result_item in tavily_crawl_results["results"]:
        log_info(
            f"TavilyCrawl: Successfully crawled {tavily_crawl_result_item['url']} from documentation site"
        )
        all_documents.append(
            Document(
                page_content=tavily_crawl_result_item["raw_content"],
                metadata={"source": tavily_crawl_result_item["url"]},
            )
        )

    # Split documents into chunks
    log_header("DOCUMENT CHUNKING PHASE")
    log_info(
        f"✂️  Text Splitter: Processing {len(all_documents)} documents with 4000 chunk size and 200 overlap",
        Colors.YELLOW,
    )
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
    splitted_docs = text_splitter.split_documents(all_documents)
    log_success(
        f"Text Splitter: Created {len(splitted_docs)} chunks from {len(all_documents)} documents"
    )

    # Process documents in batches
    indexing_ok = await index_documents(splitted_docs, batch_size=500)

    log_header("PIPELINE COMPLETE")
    if indexing_ok:
        log_success("🎉 Documentation ingestion pipeline finished successfully!")
    else:
        log_warning("Pipeline finished with errors: vectorstore indexing had failures.")
    log_info("📊 Summary:", Colors.BOLD)
    log_info(f"   • Pages crawled: {len(tavily_crawl_results)}")
    log_info(f"   • Documents extracted: {len(all_documents)}")
    log_info(f"   • Chunks created: {len(splitted_docs)}")


if __name__ == "__main__":
    asyncio.run(main())
