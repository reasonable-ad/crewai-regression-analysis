# api/server.py
"""
FastAPI server for CrewAI pipeline.

Endpoints:
- POST /analyze: Upload CSV, starts pipeline in background, returns job_id
- GET /status/{job_id}: Check if job is running/completed/error
- GET /results/{job_id}: Get the pipeline results
"""

import threading
import uuid
import json
from pathlib import Path

from fastapi import FastAPI, UploadFile, HTTPException

# Import the pipeline function
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from core.pipeline import run_pipeline


app = FastAPI(title="CrewAI Regression API")

# Directory where all jobs are stored
JOBS_DIR = Path(__file__).parent.parent / "jobs"


def save_status(job_id: str, status: str, error: str = None):
    """Save job status to status.json file."""
    status_path = JOBS_DIR / job_id / "status.json"
    data = {"status": status}
    if error:
        data["error"] = error
    with open(status_path, "w") as f:
        json.dump(data, f)


def run_job_in_background(job_id: str, csv_path: str):
    """
    Run the pipeline in a background thread.
    Updates status.json when done.
    """
    # Mark job as running
    save_status(job_id, "running")
    
    try:
        # Run the actual pipeline
        result = run_pipeline(csv_path, job_id, jobs_dir=str(JOBS_DIR))
        
        # Update status based on result
        if result["status"] == "success":
            save_status(job_id, "completed")
        else:
            save_status(job_id, "error", error=result.get("error"))
            
    except Exception as e:
        # If anything goes wrong, mark as error
        save_status(job_id, "error", error=str(e))


@app.post("/analyze")
def analyze(file: UploadFile):
    """
    Start a new analysis job.
    
    - Receives CSV file upload
    - Creates job folder with unique ID
    - Saves CSV to jobs/{job_id}/input.csv
    - Starts pipeline in background thread
    - Returns job_id immediately
    """
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Create job directory
    job_path = JOBS_DIR / job_id
    job_path.mkdir(parents=True, exist_ok=True)
    
    # Save uploaded CSV file
    csv_path = job_path / "input.csv"
    with open(csv_path, "wb") as f:
        content = file.file.read()
        f.write(content)
    
    # Initialize status as "pending"
    save_status(job_id, "pending")
    
    # Start pipeline in background thread
    thread = threading.Thread(
        target=run_job_in_background,
        args=(job_id, str(csv_path))
    )
    thread.start()
    
    # Return job_id immediately (don't wait for pipeline)
    return {"job_id": job_id}


@app.get("/status/{job_id}")
def get_status(job_id: str):
    """
    Check the status of a job.
    
    Returns:
    - {"status": "pending"} - Job created but not started
    - {"status": "running"} - Pipeline is executing
    - {"status": "completed"} - Pipeline finished successfully
    - {"status": "error", "error": "..."} - Pipeline failed
    """
    status_path = JOBS_DIR / job_id / "status.json"
    
    if not status_path.exists():
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    
    with open(status_path) as f:
        return json.load(f)


@app.get("/results/{job_id}")
def get_results(job_id: str):
    """
    Get the results of a completed job.
    
    Returns the full results including:
    - output: The crew completion text
    - logs: Execution logs
    - error: Error message if failed
    """
    results_path = JOBS_DIR / job_id / "results.json"
    
    if not results_path.exists():
        raise HTTPException(status_code=404, detail=f"Results for job {job_id} not found")
    
    with open(results_path) as f:
        return json.load(f)


# Health check endpoint
@app.get("/health")
def health():
    """Simple health check endpoint."""
    return {"status": "ok"}