# core/code_executor.py
"""
Refactored code executor for web app usage.
Key difference from notebookExecutor.py: namespace is passed during __init__
and stored, making each pipeline run isolated from others.
"""

import subprocess
import sys
import io
from contextlib import redirect_stdout
from typing import List, Optional, Type, Dict, Any

from pydantic import BaseModel, Field, PrivateAttr
from crewai.tools import BaseTool
import pandas as pd
import numpy as np


class CodeExecutorSchema(BaseModel):
    """Schema defining the input parameters for the code executor tool."""
    code: str = Field(description="The Python code to execute.")
    required_libraries: Optional[List[str]] = Field(
        default=None,
        description="A list of Python library names to install before executing the code.",
    )


class CodeExecutor(BaseTool):
    """
    Executes Python code in an isolated namespace.
    
    WHY THIS DESIGN:
    - Each pipeline run creates its own CodeExecutor with its own namespace
    - Variables created by agents (df, model, X_train, etc.) are stored in this namespace
    - After the crew finishes, we can access namespace['trained_model'] to get results
    - This prevents data leakage between concurrent users
    """
    
    name: str = "Notebook Code Executor"
    description: str = (
        "Executes Python code directly in an isolated namespace. "
        "Use this for data analysis, preprocessing, modeling, etc. "
        "Include print() statements in your code to see results. Returns stdout/stderr."
    )
    args_schema: Type[BaseModel] = CodeExecutorSchema

    # PrivateAttr: Pydantic won't try to serialize this (it's runtime-only)
    _execution_namespace: Dict[str, Any] = PrivateAttr(default_factory=dict)

    def __init__(self, namespace: Dict[str, Any] = None, **kwargs):
        """
        Initialize the code executor with an isolated namespace.
        
        Args:
            namespace: A dictionary that will hold all variables created during execution.
                      Pass an empty dict {} for each new pipeline run.
        """
        # First, let Pydantic/BaseTool do its initialization
        super().__init__(**kwargs)
        
        # Set up our execution namespace
        # If no namespace provided, create an empty one
        self._execution_namespace = namespace if namespace is not None else {}
        
        # Pre-load common libraries into the namespace so agent code can use them
        # This means agents don't need to import pandas/numpy in every code snippet
        self._execution_namespace.setdefault("pd", pd)
        self._execution_namespace.setdefault("np", np)

    def _run(self, code: str, required_libraries: Optional[List[str]] = None) -> str:
        """
        Execute Python code in the isolated namespace.
        
        This is called by CrewAI when an agent uses this tool.
        
        Args:
            code: Python code string to execute
            required_libraries: Optional list of pip packages to install first
            
        Returns:
            String containing installation log + execution output/errors
        """
        installation_log = ""
        
        # --- Step 1: Install any required libraries ---
        if required_libraries:
            installation_log += "--- Installing Libraries ---\n"
            python_executable = sys.executable
            
            for lib in required_libraries:
                installation_log += f"Attempting to install {lib}...\n"
                try:
                    process = subprocess.run(
                        [python_executable, "-m", "pip", "install", lib],
                        capture_output=True,
                        text=True,
                        check=False,
                        timeout=120,
                    )
                    if process.returncode == 0:
                        installation_log += f"Successfully installed {lib}.\n"
                    else:
                        installation_log += (
                            f"Failed to install {lib}. "
                            f"RetCode: {process.returncode}\n"
                            f"Stderr: {process.stderr}\n"
                        )
                except Exception as e:
                    installation_log += f"Error installing {lib}: {e}\n"
                    
            installation_log += "--- Library Installation Finished ---\n\n"

        # --- Step 2: Execute the code ---
        execution_log = "--- Executing Code ---\n"
        output_buffer = io.StringIO()
        
        try:
            # redirect_stdout captures print() statements
            with redirect_stdout(output_buffer):
                # exec() runs the code string
                # The namespace dict stores all variables created by the code
                # e.g., if code does "model = RandomForest()", 
                #       then self._execution_namespace['model'] = <the model>
                exec(code, self._execution_namespace)

            execution_output = output_buffer.getvalue()
            execution_log += (
                f"Code executed successfully. Output:\n"
                f"```output\n{execution_output or '[No Print Output]'}\n```\n"
            )
            return installation_log + execution_log

        except Exception as e:
            # Capture any errors that occurred during execution
            error_message = f"Error executing code: {type(e).__name__}: {e}\n"
            partial_output = output_buffer.getvalue()
            
            if partial_output:
                error_message += (
                    f"Captured output before error:\n"
                    f"```output\n{partial_output}\n```\n"
                )
                
            execution_log += error_message
            return installation_log + execution_log

    def get_namespace(self) -> Dict[str, Any]:
        """
        Get the execution namespace to access variables created during execution.
        
        WHY THIS EXISTS:
        After the crew finishes, call this to get the namespace dict.
        You can then access namespace['trained_model'], namespace['X_test'], etc.
        to extract results without parsing text output.
        
        Returns:
            The namespace dictionary containing all variables
        """
        return self._execution_namespace
