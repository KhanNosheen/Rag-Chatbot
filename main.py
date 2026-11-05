import os
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise RuntimeError("GOOGLE_API_KEY is missing from .env file.")

app = FastAPI(title="RAG AI Integration API")

# Global variables to maintain session/state in memory
vectorstore = None
rag_chain = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str

@app.get("/")
def read_root():
    return {"message": "RAG Backend API is running successfully!"}

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    global vectorstore, rag_chain

    if not (file.filename.endswith(".pdf") or file.filename.endswith(".txt")):
        raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")

    try:
        suffix = f".{file.filename.split('.')[-1]}"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_filepath = tmp_file.name

        if file.filename.endswith(".pdf"):
            loader = PyPDFLoader(tmp_filepath)
        else:
            loader = TextLoader(tmp_filepath, encoding="utf-8")
        
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        splits = text_splitter.split_documents(docs)

        # Initialize Google GenAI Embeddings
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001",
            google_api_key=api_key
        )
        vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)

        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Initialize Google GenAI Chat Model
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.8-flash",
            google_api_key=api_key,
            temperature=0.3
        )

        system_prompt = (
            "You are an assistant for answering questions using the provided document context. "
            "Use the following retrieved context to answer the user's question clearly. "
            "If the answer is not in the context, state 'I cannot find that in the provided document.'\n\n"
            "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        os.remove(tmp_filepath)

        return {
            "status": "success",
            "filename": file.filename,
            "chunks_indexed": len(splits),
            "message": "Document indexed successfully."
        }

    except Exception as e:
        print(f"UPLOAD ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload processing failed: {str(e)}")

@app.post("/chat", response_model=QueryResponse)
async def chat_with_document(request: QueryRequest):
    global rag_chain

    if rag_chain is None:
        raise HTTPException(
            status_code=400, 
            detail="No document loaded. Please upload a file via /upload first."
        )

    try:
        response = rag_chain.invoke({"input": request.question})
        
        # Safely parse the response output
        answer_text = response.get("answer") or response.get("text") or str(response)
        
        return QueryResponse(
            question=request.question,
            answer=answer_text
        )
    except Exception as e:
        print(f"CHAT ERROR DETAIL: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat generation failed: {str(e)}")