#!/bin/bash
# start.sh
# Starts both FastAPI and Streamlit in the same container

# Start FastAPI in background
echo "Starting FastAPI server..."
uvicorn api.server:app --host 0.0.0.0 --port 8000 &

# Start Streamlit in foreground
echo "Starting Streamlit app..."
streamlit run app/streamlit_app.py --server.port 8501 --server.address 0.0.0.0
