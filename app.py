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

model = SentenceTransformer('msmarco-bert-base-dot-v5')
index = None
text_chunks = []
metadata = []

# Chatbot logic (simple example)
class ChatbotRequest(BaseModel):
    query: str

@app.post("/chatbot/")
async def chatbot_endpoint(request: ChatbotRequest):
    query = request.query

    # Ensure the index and text_chunks are loaded
    if index is None or not text_chunks:
        raise HTTPException(status_code=400, detail="No document uploaded or processed yet.")

    # Step 1: Break down the query into smaller sub-queries using the LLM
    sub_queries = decompose_query(query)

    # Step 2: Query FAISS for each sub-query and collect the context
    context = query_faiss_for_sub_queries(sub_queries)

    # Step 3: Send the query and combined context to Ollama for the final response
    response = ollama_chat(query, context)

    return {"response": response}


# Function to decompose the query into smaller sub-queries using an LLM
def decompose_query(query):
    prompt = f"""
    You are a question decomposition assistant. Please break down the following query into smaller, more focused sub-queries:
    Query: {query}
    """
    response = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False})

    try:
        sub_queries = response.json().get("response", "").split("\n")
        return [q.strip() for q in sub_queries if q.strip()]
    except requests.exceptions.JSONDecodeError:
        return [query]  # Fallback: Use the original query if decomposition fails


# Function to query FAISS for each sub-query
def query_faiss_for_sub_queries(sub_queries):
    global metadata  # Access global metadata

    context = []
    for sub_query in sub_queries:
        query_embedding = model.encode([sub_query])
        distances, indices = index.search(np.array(query_embedding), k=5)

        # Debug: Print indices to see the values returned by FAISS
        print(f"Indices returned by FAISS: {indices[0]}")

        # Prepare a list of contexts with document name and page number
        result_context = []
        for idx in indices[0]:
            if idx < len(metadata):  # Check if the index is within bounds
                doc_metadata = metadata[idx]  # Retrieve metadata for the chunk
                result_context.append(f"Document: {doc_metadata['document_name']}, Page: {doc_metadata['page_number']}\nContent: {doc_metadata['chunk_text']}")
            else:
                # Handle invalid index (this should not happen in normal cases)
                result_context.append(f"Invalid index: {idx}")

        # Combine the context from all sub-queries
        context.append("\n\n".join(result_context))

    return "\n\n".join(context)


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
    global index, text_chunks, metadata  # Access global variables

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

    # Store metadata with chunk text: (document_name, page_number, text_chunk)
    metadata = []
    for i, page in enumerate(pages):
        for chunk in text_chunks[i::len(pages)]:  # Ensure chunk to page mapping
            metadata.append({
                'document_name': file_path,  # Store the file name
                'page_number': i + 1,  # Page number starts from 1
                'chunk_text': chunk
            })

    # Generate embeddings for the text chunks
    embeddings = model.encode([m['chunk_text'] for m in metadata])

    # Convert embeddings to a NumPy array
    embeddings_np = np.array(embeddings)

    # Initialize FAISS index for L2 (Euclidean) distance search
    index = faiss.IndexFlatL2(embeddings_np.shape[1])
    index.add(embeddings_np)  # Add the embeddings to the FAISS index

    print("Embeddings added to the FAISS index")


def ollama_chat(query, context):
    prompt_qa = """
Given the following context:

{context}

Please provide a clear, concise, and accurate answer to the following question:

Question: {query}

Guidelines:
- Only use the information provided in the context.
- If the context does not provide enough information to answer the question, respond with: "Insufficient information provided to answer the query."
- Replace any mention of the word 'Kunai' with 'The company'.
- Ensure the response is relevant and to the point, avoiding unnecessary details.
- At the **end** of your answer, append **[Employee Handbook.pdf]** to the response.
""".format(context=context, query=query)

    data = {
        "model": "llama3.2",  # Specify the model you are using in Ollama
        "prompt": prompt_qa,
        "stream": False,
        "format": {
            "type": "object",
            "properties": {
                "response": {
                    "type": "string"
                }
            },
            "required": ["response"]
         }
    }

    # Send a POST request to the Ollama API
    response = requests.post('http://localhost:11434/api/generate', json=data)

    try:
        return response.json().get('response', 'No response from Ollama')
    except requests.exceptions.JSONDecodeError:
        return f"Failed to decode JSON response: {response.text}"

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
