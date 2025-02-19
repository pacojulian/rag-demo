#!/bin/bash

# Kill the Ollama server
echo "Stopping Ollama server..."
kill $OLLAMA_PID
echo "Ollama server stopped."

# Kill the Next.js app
echo "Stopping Next.js application..."
kill $NEXTJS_PID
echo "Next.js application stopped."

# Kill the FastAPI application
echo "Stopping FastAPI application..."
kill $FASTAPI_PID
echo "FastAPI application stopped."

echo "All processes stopped."

