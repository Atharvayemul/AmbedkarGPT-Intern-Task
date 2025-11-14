## Phase 1 – Core Skills Evaluation

A simple command-line RAG-based Q&A system built using LangChain, HuggingFace Embeddings, ChromaDB, and Ollama (Mistral 7B).

🚀 Project Description

This project is a functional prototype demonstrating a Retrieval-Augmented Generation (RAG) pipeline.
The system:

- Loads a speech text file (speech.txt)

- Splits the data into manageable chunks

- Creates embeddings using sentence-transformers/all-MiniLM-L6-v2

- Stores embeddings locally in ChromaDB

- Retrieves relevant chunks for a user’s question

- Uses Ollama Mistral 7B to answer based on retrieved context

- Accepts questions directly from the command line

