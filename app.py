import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

import time

def invoke_model_with_retry(chain_or_model_call, input_data, max_retries=3, initial_delay=3):
    """
    Safely executes a Google API call with automatic retries on 503 / Service Unavailable errors.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return chain_or_model_call(input_data)
        except Exception as e:
            # Check if it's a 503 error or high demand message
            error_str = str(e)
            if "503" in error_str or "UNAVAILABLE" in error_str or "high demand" in error_str:
                if attempt == max_retries:
                    st.error(f"Google API is experiencing high demand and failed after {max_retries} attempts. Please try again later.")
                    raise e
                
                wait_time = initial_delay * attempt
                st.warning(f"Google API busy (503). Retrying attempt {attempt} of {max_retries} in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                # If it's a different kind of error, raise it immediately
                raise e

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="RAG Document Q&A Assistant", layout="wide")
st.title("📄 Document Q&A RAG Assistant (Powered by Gemini & ChromaDB)")

if not api_key:
    st.error("Please add your GOOGLE_API_KEY to the .env file.")
    st.stop()

# Initialize session state for vectorstore and chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar for document processing
with st.sidebar:
    st.header("1. Upload Knowledge Base")
    uploaded_file = st.file_uploader("Upload a PDF or TXT document", type=["pdf", "txt"])
    
    if uploaded_file and st.button("Process Document"):
        with st.spinner("Processing & indexing document..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=uploaded_file.name) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_filepath = tmp_file.name

            if uploaded_file.name.endswith(".pdf"):
                loader = PyPDFLoader(tmp_filepath)
            else:
                loader = TextLoader(tmp_filepath)
            
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            splits = text_splitter.split_documents(docs)

            embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
            
            st.session_state.vectorstore = vectorstore
            st.success(f"Document processed! {len(splits)} chunks indexed into ChromaDB.")

# Main Q&A Interface
st.header("2. Ask Questions About Your Document")

if st.session_state.vectorstore is None:
    st.info("Upload and process a document from the sidebar to begin querying.")
else:
    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", temperature=0.2)

    system_prompt = (
        "You are an assistant for answering questions using the provided document context. "
        "Use the following retrieved context to answer the user's question clearly. "
        "If the answer is not in the context, explicitly state 'I cannot find that in the provided document.'\n\n"
        "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    user_query = st.chat_input("Ask a question about the uploaded document...")

    if user_query:
        st.session_state.chat_history.append({"role": "user", "content": user_query})

        with st.spinner("Retrieving context and generating answer..."):
            # Wrapped with our retry protection function
            response = invoke_model_with_retry(
                lambda query: rag_chain.invoke({"input": query}), 
                user_query
            )
            answer = response["answer"]

        st.session_state.chat_history.append({"role": "assistant", "content": answer})

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])