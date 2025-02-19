import os
import shutil
import requests
import numpy as np
import faiss
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi.middleware.cors import CORSMiddleware  # Import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins, change to specific domains if needed
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)
# Initialize global variables for FAISS index and model
model = SentenceTransformer('all-MiniLM-L6-v2')
index = None
text_chunks = []


# Chatbot logic (simple example)
class ChatbotRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot_endpoint(request: ChatbotRequest):
    query = request.query

    # Ensure the index and text_chunks are loaded
    if index is None or not text_chunks:
        raise HTTPException(status_code=400, detail="No document uploaded or processed yet.")

    # Query the document using the FAISS index and model
    response = query_pdf(index, text_chunks, query, model)

    return {"response": response}


# Document upload endpoint
@app.post("/upload_document/")
async def upload_document(file: UploadFile = File(...)):
    # Only accept PDF or DOCX files
    if not (file.filename.endswith(".pdf") or file.filename.endswith(".docx")):
        raise HTTPException(status_code=400, detail="Only .pdf or .docx files are allowed")

    # Save the file temporarily
    temp_file_path = save_file_locally(file)

    # Process the uploaded PDF
    await load_and_process_pdf(temp_file_path)

    # Remove the temporary file after processing
    os.remove(temp_file_path)

    return {"info": f"File '{file.filename}' has been uploaded and processed successfully."}


def save_file_locally(uploaded_file: UploadFile):
    temp_file_path = f"temp_{uploaded_file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(uploaded_file.file, buffer)
    return temp_file_path


# Async function to process and load PDF files
async def load_and_process_pdf(file_path: str):
    global index, text_chunks  # Access global variables

    # Load the PDF using PyPDFLoader with the file path
    loader = PyPDFLoader(file_path)
    pages = []

    # Load pages asynchronously
    async for page in loader.alazy_load():
        pages.append(page)

    # Combine the page contents into a single text
    full_text = " ".join([page.page_content for page in pages])

    # Split text into chunks using RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks = splitter.split_text(full_text)

    # Generate embeddings for the text chunks
    embeddings = model.encode(text_chunks)

    # Convert embeddings to a NumPy array
    embeddings_np = np.array(embeddings)

    # Initialize FAISS index for L2 (Euclidean) distance search
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)  # Add the embeddings to the FAISS index

    print("Embeddings added to the FAISS index")


def query_pdf(index, text_chunks, query, model):
    # Encode the query to get its embedding
    query_embedding = model.encode([query])

    # Search the FAISS index for similar text chunks
    distances, indices = index.search(np.array(query_embedding), k=5)

    # Retrieve the relevant chunks from the indices
    context = "\n\n".join([text_chunks[idx] for idx in indices[0]])

    # Send the query and context to Ollama
    ollama_response = ollama_chat(query, context)
    print(f"Ollama's response: {ollama_response}")

    return ollama_response


def ollama_chat(query, context):
    headers = {
        "Content-Type": "application/json",
    }

    data = {
        "model": "llama3.2",  # Specify the model you are using in Ollama
        "prompt": f"{context}\n{query}",  # Adjust the prompt as needed
        "stream": False,  # Setting stream to false as in the curl command
         "format": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string"
                }
            },
            "required": [
                "response"
            ]
         }
    }

    # Send a POST request to the Ollama API
    response = requests.post('http://localhost:11434/api/generate', json=data, headers=headers)

    try:
        # Try to return the 'response' field from the JSON response
        return response.json().get('response', 'No response from Ollama')
    except requests.exceptions.JSONDecodeError:
        # Return an error message if JSON decoding fails
        return f"Failed to decode JSON response: {response.text}"


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
