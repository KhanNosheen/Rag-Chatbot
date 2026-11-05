# 🤖 RAG-Based Document Q&A Chatbot

An intelligent Retrieval-Augmented Generation (RAG) chatbot built with Python, LangChain, Google Gemini, and ChromaDB. This application allows users to upload custom documents (PDF, TXT) and interactively query their content through a clean Streamlit interface powered by state-of-the-art vector search and LLM generation.

🚀 **Live Demo:** [Check out the live Streamlit App](https://rag-chatbot-si8ritpccgygeurzvngqyu.streamlit.app/)

---

## 🛠️ Tech Stack & Tools

* **Core Language:** Python 3.11+
* **Frontend & Deployment:** Streamlit & Streamlit Community Cloud
* **Orchestration & RAG Framework:** LangChain (`langchain-community`, `langchain-google-genai`)
* **Embeddings & LLM:** Google Gemini API (`gemini-1.5-flash`)
* **Vector Database:** ChromaDB
* **Document Parsing:** PyPDF / LangChain Document Loaders

---

## ⚙️ Key Features

* **Document Ingestion:** Supports uploading and parsing multi-page PDF and text files.
* **Vector Embeddings & Storage:** Efficient chunking and embedding storage using ChromaDB for high-speed semantic retrieval.
* **Context-Aware Responses:** Queries the stored vector database to fetch relevant text chunks and feeds them as grounded context to Google Gemini.
* **Secure Secret Management:** Uses Streamlit Secrets (`st.secrets`) for secure API key injection during cloud deployment.

---

## 📂 Project Structure

```text
Rag-Chatbot/
│
├── app.py              # Main Streamlit application script
├── requirements.txt    # Project dependencies for cloud reproduction
└── README.md           # Project documentation