# app/streamlit_app.py
"""
Streamlit frontend for CrewAI Regression Analysis.

Features:
- Upload CSV file
- Run analysis button
- Poll for status while running
- Display results when complete
- Download logs
"""

import streamlit as st
import requests
import time

# API URL - change this if running on different host/port
API_URL = "http://localhost:8000"


def main():
    st.title("CrewAI Regression Analysis")
    st.write("Upload a CSV file with a 'Target' column to run regression analysis.")
    
    # File upload widget
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type="csv",
        help="Upload a CSV file. The target column should be named 'Target'."
    )
    
    # Show file preview if uploaded
    if uploaded_file is not None:
        st.write("**File uploaded:**", uploaded_file.name)
    
    # Run Analysis button
    if uploaded_file is not None:
        if st.button("Run Analysis", type="primary"):
            run_analysis(uploaded_file)


def run_analysis(uploaded_file):
    """Send file to API and poll for results."""
    
    # Step 1: Send file to API
    st.write("Sending file to API...")
    
    try:
        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "text/csv")}
        response = requests.post(f"{API_URL}/analyze", files=files)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        st.error("Could not connect to API. Make sure the FastAPI server is running on http://localhost:8000")
        return
    except Exception as e:
        st.error(f"Error sending file: {e}")
        return
    
    job_id = response.json()["job_id"]
    st.write(f"Job started with ID: `{job_id}`")
    
    # Step 2: Poll for status
    status_placeholder = st.empty()
    
    while True:
        try:
            status_response = requests.get(f"{API_URL}/status/{job_id}")
            status_data = status_response.json()
            status = status_data["status"]
        except Exception as e:
            st.error(f"Error checking status: {e}")
            return
        
        if status == "completed":
            status_placeholder.success("Analysis completed!")
            break
        elif status == "error":
            error_msg = status_data.get("error", "Unknown error")
            status_placeholder.error(f"Analysis failed: {error_msg}")
            break
        else:
            # Still running
            status_placeholder.info(f"Status: {status}... (checking every 5 seconds)")
            time.sleep(5)
    
    # Step 3: Get and display results
    if status == "completed":
        display_results(job_id)


def display_results(job_id: str):
    """Fetch and display results from API."""
    
    try:
        response = requests.get(f"{API_URL}/results/{job_id}")
        response.raise_for_status()
        results = response.json()
    except Exception as e:
        st.error(f"Error fetching results: {e}")
        return
    
    # Display the crew output
    st.header("Results")
    
    if results.get("output"):
        st.code(results["output"])
    else:
        st.warning("No output available")
    
    # Display error if any
    if results.get("error"):
        st.error(f"Error: {results['error']}")
    
    # Download buttons
    st.header("Downloads")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if results.get("logs"):
            st.download_button(
                label="Download Logs",
                data=results["logs"],
                file_name=f"logs_{job_id}.txt",
                mime="text/plain"
            )
    
    with col2:
        if results.get("output"):
            st.download_button(
                label="Download Results",
                data=results["output"],
                file_name=f"results_{job_id}.txt",
                mime="text/plain"
            )


if __name__ == "__main__":
    main()