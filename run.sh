#!/bin/bash

# Start Ollama server
echo "Starting Ollama server..."
ollama run llama3.2 &
OLLAMA_PID=$!
echo "Ollama server started with PID $OLLAMA_PID"

# Switch to Node.js 20 and run Next.js app
echo "Switching to Node.js 20 for Next.js app..."
pushd document-chat-app && pnpm dev &
NEXTJS_PID=$!
echo "Next.js app started with PID $NEXTJS_PID"
popd  # Return to the original directory

# Start FastAPI application
echo "Starting FastAPI application..."
source venv/bin/activate && uvicorn app:app --reload &
FASTAPI_PID=$!
echo "FastAPI app started with PID $FASTAPI_PID"

# Wait for all processes to complete (so the script doesn't exit)
wait

