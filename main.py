
import os
import re
import time
import fitz
from bs4 import BeautifulSoup
from azure.storage.blob import BlobServiceClient
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from pinecone import Pinecone
from fastapi import FastAPI
from pydantic import BaseModel
from dotenv import load_dotenv


# ================= CONFIG =================

load_dotenv() 
          
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = "ragbot"
AZURE_STORAGE_CONNECTION_STRING = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
AZURE_CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER_NAME")

# ================= FASTAPI =================
app = FastAPI(title="RAG Support Bot API")

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    latency_ms: float

# ================= LLM =================
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0,
    max_tokens=500,
    api_key=OPENAI_API_KEY
)

def generate_answer(prompt: str) -> str:
    return llm.invoke(prompt).content

# ================= EMBEDDINGS =================
embeddings = OpenAIEmbeddings(
    model="text-embedding-ada-002",
    openai_api_key=OPENAI_API_KEY
)

# ================= UTILS =================
def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def extract_text_from_html(html_bytes):
    soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return clean_text(soup.get_text(separator=" "))

# ================= LOAD DOCS =================
def load_documents_from_azure():
    blob_service = BlobServiceClient.from_connection_string(
        AZURE_STORAGE_CONNECTION_STRING
    )
    container_client = blob_service.get_container_client(
        AZURE_CONTAINER_NAME
    )

    documents = []
    for blob in container_client.list_blobs():
        blob_client = container_client.get_blob_client(blob.name)
        file_bytes = blob_client.download_blob().readall()

        if blob.name.lower().endswith(".pdf"):
            text = ""
            with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                for page in doc:
                    text += page.get_text()

            documents.append({
                "text": text,
                "metadata": {
                    "document_name": blob.name,
                    "document_type": "policy"
                }
            })

        elif blob.name.lower().endswith((".html", ".htm")):
            documents.append({
                "text": extract_text_from_html(file_bytes),
                "metadata": {
                    "document_name": blob.name,
                    "document_type": "faq"
                }
            })
    return documents

# ================= CHUNKING =================
def chunk_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = []
    for doc in documents:
        for chunk in splitter.split_text(clean_text(doc["text"])):
            chunks.append({
                "text": chunk,
                "metadata": doc["metadata"]
            })
    return chunks

# ================= PINECONE =================
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)

def store_in_pinecone(chunks):
    vectors = []
    for i, chunk in enumerate(chunks):
        vectors.append((
            f"chunk-{i}",
            embeddings.embed_query(chunk["text"]),
            {**chunk["metadata"], "text": chunk["text"]}
        ))
    index.upsert(vectors)

def retrieve_context(query: str, top_k: int = 3):
    return index.query(
        vector=embeddings.embed_query(query),
        top_k=top_k,
        include_metadata=True
    )["matches"]

# ================= PROMPT =================
def build_prompt(matches, question: str) -> str:
    context = "\n\n".join(m["metadata"]["text"][:700] for m in matches)
    return f"""
Answer ONLY using the context below.
If the answer is not present, say you do not know.

Context:
{context}

Question:
{question}
"""

# ================= API =================
@app.post("/ask", response_model=AnswerResponse)
def ask_question(request: QuestionRequest):
    start = time.time()
    matches = retrieve_context(request.question)
    prompt = build_prompt(matches, request.question)
    answer = generate_answer(prompt)
    latency = round((time.time() - start) * 1000, 2)

    return {"answer": answer, "latency_ms": latency}

@app.get("/")
def health():
    return {"status": "RAG API running"}

# ================= INGEST (ON STARTUP) =================
docs = load_documents_from_azure()
chunks = chunk_documents(docs)
store_in_pinecone(chunks)


