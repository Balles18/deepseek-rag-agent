📄 RAG System with DeepSeek R1 & Ollama
This project demonstrates how to build a Retrieval-Augmented Generation (RAG) system using DeepSeek R1 (via Ollama) as the LLM, FAISS for vector storage, and Hugging Face embeddings for semantic understanding of PDF documents. The app is built with Streamlit for a user-friendly interface.

🚀 Features
Upload a PDF document and ask questions based on its content.

Automatically split the document into semantically meaningful chunks.

Embed the document using Hugging Face models.

Perform semantic similarity search with FAISS.

Use DeepSeek R1 via Ollama to generate answers to user questions.

Simple and modern Streamlit interface with custom styling.

🧱 Tech Stack
Tool / Library	Purpose
Streamlit	Web App Frontend
PDFPlumberLoader	PDF Parsing
SemanticChunker	Semantic Chunking of Text
Hugging Face Embeddings	Text Embeddings for Vector Search
FAISS	Vector Search and Similarity Retrieval
Ollama (DeepSeek R1)	Large Language Model (LLM) for Q&A
LangChain	Orchestrating Components




Example Commands
Upload a PDF (e.g., a research paper, report, or article).

Ask questions like:

"Summarize the main findings."

"What are the key points in section 3?"

"Explain the conclusion in simple terms."