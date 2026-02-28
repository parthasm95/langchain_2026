# Document Helper Project

## Overview
The Document Helper project aims to streamline the process of querying documentation using large language models (LLMs). This tool allows users to easily access and query documentation for specific packages, providing examples and FAQs in a user-friendly manner.

## Objectives
- Ingest and index documentation into a vector store
- Enable flexible querying capabilities using LLMs
- Create a user interface with Streamlit for easy interaction

## Tools and Technologies
- **LangChain**: For creating the querying pipeline
- **tavily**: For downloading documentation
- **tavily crawl**: Upgraded tool for documentation ingestion and indexing
- **Streamlit**: For building the front-end user interface

## Project Phases

1. **Documentation Ingestion**:  
   - Download the latest version of the desired documentation.
   - Convert the documentation pages into vector embeddings and store them in a vector database.

2. **Chain Writing**:  
   - Develop a chain that utilizes the vector database to fetch relevant chunks of information based on user queries.

3. **User Interface Development**:  
   - Implement a Streamlit-based UI for smooth interaction with the tool.

4. **Memory Integration**:  
   - Enhance the chat capabilities with memory features so it can recall past interactions.

## Getting Started
1. Clone the repository.
2. Install the necessary dependencies.
3. Follow the instructions to set up the tools.
4. Begin the ingestion process and start querying the documentation.

## Future Enhancements
- Explore how to handle various document formats such as PDFs.
- Improve the querying functionalities with more advanced NLP techniques.

