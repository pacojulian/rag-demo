import asyncio
import faiss
import numpy as np
import requests
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

def ollama_chat(query, context):
    print(f"Context generated: {context}")  # Ensure context is being printed

    if not context.strip():  # Check if context is empty
        return "Document not processed or insufficient embeddings. Cannot provide an answer."

    initial_prompt = """
        Given the context information provided below, and without using any prior knowledge, answer the query.
        Context: {context}
        Query: {query}
        If you are unable to answer the query based on the context, respond with "Insufficient information provided to answer the query."
    """.format(context=context, query=query)

    headers = {
        "Content-Type": "application/json",
    }
    data = {
        "model": "deepseek-r1",  # Specify the model you are using in Ollama
        "prompt": initial_prompt,
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

    response = requests.post('http://localhost:11434/api/generate', json=data, headers=headers)

    try:
        return response.json().get('response', 'No response from Ollama')
    except requests.exceptions.JSONDecodeError:
        return f"Failed to decode JSON response: {response.text}"

# Async function to load PDF and process it
async def load_and_process_pdf():
    file_path = "./safety.pdf"
    loader = PyPDFLoader(file_path)
    pages = []

    async for page in loader.alazy_load():
        pages.append(page)

    full_text = " ".join([page.page_content for page in pages])

    if not full_text.strip():
        return None, None  # Return None if the PDF is empty or not processed

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
    text_chunks = splitter.split_text(full_text)

    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(text_chunks)

    embeddings_np = np.array(embeddings)

    index = faiss.IndexFlatL2(embeddings_np.shape[1])  # L2 distance for similarity
    index.add(embeddings_np)  # Add embeddings to the index

    print("PDF loaded and indexed successfully!")
    print(f"Number of text chunks generated: {len(text_chunks)}")  # Debugging number of chunks
    return index, text_chunks

def query_pdf(index, text_chunks, query, model):
    if index is None or text_chunks is None:  # Check if PDF is not processed
        print("Document not processed or no embeddings found.")
        return

    query_embedding = model.encode([query])

    distances, indices = index.search(np.array(query_embedding), k=5)

    context = "\n\n".join([text_chunks[idx] for idx in indices[0]])

    # Print out the selected chunks for debugging
    print(f"Selected chunks for context: {[text_chunks[idx] for idx in indices[0]]}")

    ollama_response = ollama_chat(query, context)
    print(f"Ollama's response: {ollama_response}")

async def main():
    index, text_chunks = await load_and_process_pdf()

    model = SentenceTransformer('all-MiniLM-L6-v2')
    if index is None or text_chunks is None:  # If the document wasn't processed properly
        print("Failed to load and process the document. Please check the document or embeddings.")
        return

    while True:
        query = input("Ask your PDF: ")
        if query.lower() in ['exit', 'quit']:
            break
        query_pdf(index, text_chunks, query, model)

asyncio.run(main())
