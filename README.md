# 🚀 EgSA Unified AI Copilot

## 📌 Overview
An End-to-End Secure Generative AI Copilot proposed for the **Egyptian Space Agency (EgSA)**. This system utilizes advanced **RAG (Retrieval-Augmented Generation)** combined with a highly efficient, locally-hosted Large Language Model (Qwen 1.5B with 4-bit Quantization). It allows space engineers to securely query satellite manuals, space laws, and research papers (e.g., EgyptSat-1 performance studies) in both Arabic and English.

## ✨ Key Features
- **🔐 Secure Authentication:** Built-in login system ensuring only authorized engineers can access sensitive data.
- **🧠 Advanced RAG Architecture:** Employs `ChromaDB` for vector storage and `multilingual-e5-small` for cross-lingual semantic embeddings.
- **⚡ Local LLM Execution:** Runs entirely on a local GPU using `BitsAndBytes` 4-bit quantization, guaranteeing 100% Data Privacy (no external APIs).
- **🎨 Interactive UI:** A clean, professional web interface built with `Streamlit`.

## 🛠️ Tech Stack
- **AI/ML:** PyTorch, Transformers, HuggingFace
- **RAG Engine:** LangChain, ChromaDB, Sentence-Transformers
- **Frontend:** Streamlit

## 👨‍💻 Developed By
**Ahmed Ehab** 
AI & Machine Learning Engineering Student