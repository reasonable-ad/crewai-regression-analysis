# core/pipeline.py
"""
Main CrewAI pipeline module.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from crewai import LLM, Agent, Task, Crew, Process
from core.code_executor import CodeExecutor


def load_config() -> str:
    """Load environment variables and return the HuggingFace API key."""
    load_dotenv()
    
    api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if not api_key:
        raise ValueError(
            "HUGGINGFACEHUB_API_TOKEN not found in environment variables. "
            "Make sure you have a .env file with this key."
        )
    
    return api_key


def create_llm() -> LLM:
    """Create and configure the HuggingFace LLM."""
    api_key = load_config()
    
    llm = LLM(
        model="huggingface/Qwen/Qwen2.5-Coder-32B-Instruct",
        api_key=api_key,
        temperature=0.1
    )
    
    return llm


def create_agents(llm: LLM, executor_tool: CodeExecutor) -> Dict[str, Agent]:
    """Create the three agents for the pipeline."""
    
    # EXACT text from your notebook (only 'Units Sold' changed to 'Target')
    planner_agent = Agent(
        role="Lead Data Scientest and Planner",
        goal=(
            "Analyze the objective (predict 'Target') assuming data is in a global pandas DataFrame 'shared_df'. "
            "Create a step-by-step plan for regression analysis. Instruct subsequent agents on the GOALS for each step."
            "(e.g., inspect data, preprocess, model, evaluate) and tell them to use the 'Notebook Code Executor' tool "
            "to WRITE and EXECUTE the necessary Python code."
        ),
        backstory=(
            "Experienced data scientist planning ML projects. Knows data is in 'shared_df' and agents will write and execute code using a tool."
        ),
        llm=llm,
        allow_delegation=False,
        verbose=True
    )
    
    # EXACT text from your notebook (only 'Units Sold' changed to 'Target')
    analyst_preprocessor_agent = Agent(
        role="Data Analysis and Preprocessing Expert",
        goal=(
            "Follow the plan for exploratory data analysis and preprocessing. You will work with the global pandas DataFrame 'shared_df' where the target column is named 'Target'. "
        "Your job has two main parts: "
        "PART 1 - EXPLORATORY DATA ANALYSIS: Write Python code to thoroughly inspect the dataset. "
        "Print the shape, column names and data types using info(), check for missing values using isnull().sum(), "
        "print statistical summaries using describe(), identify which columns are numeric, which are categorical (object dtype), "
        "and which might be date columns or identifier columns (like IDs, names, codes). "
        "PART 2 - PREPROCESSING: Based on your EDA findings, perform appropriate preprocessing. "
        "For each preprocessing step, print a message explaining WHY you are doing it. "
        "Handle missing values if any exist (drop rows, fill with mean/median/mode depending on column type). "
        "Handle outliers in numeric columns using capping (clip values to 1st and 99th percentiles or IQR method). "
        "If date columns exist, convert them to datetime format, optionally extract useful features (year, month, day), then drop the original date column. "
        "Drop identifier columns that should not be used as features (columns containing IDs, names, product codes, etc.). "
        "For categorical columns with low cardinality, apply One-Hot Encoding using pd.get_dummies(). Update 'shared_df' with preprocessed data. "
        "Apply standardization (StandardScaler) on numeric feature columns if you find it necessary based on the data distribution. "
        "Create global variable 'y' containing the 'Target' column. Create global variable 'X' containing all other columns after dropping 'Target'. "
        "Split into global variables 'X_train', 'X_test', 'y_train', 'y_test' using 80/20 split. Print the shapes of all created variables to confirm success. "
        "Use the 'Notebook Code Executor' tool to execute ALL your code. "
        "IMPORTANT: You MUST call the 'Notebook Code Executor' tool to run your code. DO NOT just write code in your response - actually CALL the tool. Your task FAILS if you do not invoke the tool."
        ),
        backstory=(
            "Meticulous data analyst skilled in writing pandas and scikit-learn code. Expert at inspecting datasets and making data-driven preprocessing decisions. "
        "Always explains reasoning for each preprocessing step taken. Uses the 'Notebook Code Executor' tool to run generated code. "
        "Knows data is in global 'shared_df' with target column 'Target' and must create global train/test variables for the modeling agent to use."
        ),
        llm=llm,
        tools=[executor_tool],
        allow_delegation=False,
        verbose=True
    )
    
    # EXACT text from your notebook
    modeler_evaluator_agent = Agent(
        role="Machine Learning Modeler and Evaluator",
        goal=(
            "Follow the plan for modeling and evaluation. You will train multiple regression models, evaluate them all, and identify the best performing model for each metric. "
        "Assume global variables X_train, X_test, y_train, y_test already exist from the previous agent. "
        "Your job is to: "
        "1. Train exactly 5 different regression models: LinearRegression, DecisionTreeRegressor (random_state=42), RandomForestRegressor (random_state=42, n_estimators=100), GradientBoostingRegressor (random_state=42), and Ridge (alpha=1.0). "
        "2. For EACH model: fit on X_train/y_train, predict on X_test, calculate these metrics: MAE (Mean Absolute Error), MSE (Mean Squared Error), RMSE (Root Mean Squared Error), R² (R-squared score). "
        "3. Store all results in a comparison DataFrame with columns: Model, MAE, MSE, RMSE, R2. Print the full comparison table. "
        "4. Identify and print which model has the BEST (lowest) MAE, which has the BEST (lowest) RMSE, and which has the BEST (highest) R². "
        "5. For the best overall model (based on R²), calculate and print the top 10 feature importances if the model supports it (tree-based models have feature_importances_). "
        "If the best model doesn't support feature importances, use the RandomForestRegressor's feature importances instead. "
        "6. Provide a final recommendation explaining which model you recommend and why, noting any trade-offs between metrics. "
        "Use the 'Notebook Code Executor' tool to execute ALL your code. "
        "IMPORTANT: You MUST call the 'Notebook Code Executor' tool to run your code. DO NOT just write code in your response - actually CALL the tool. Your task FAILS if you do not invoke the tool."
        ),
        backstory=(
            "ML engineer specialized in regression modeling who believes in comparing multiple models before making recommendations. "
        "Writes clean scikit-learn code and uses the 'Notebook Code Executor' tool to run it. "
        "Understands that different models may excel at different metrics and always provides clear explanations of trade-offs. "
        "Expects global train/test split variables (X_train, X_test, y_train, y_test) to be available from the preprocessing agent."
        ),
        llm=llm,
        tools=[executor_tool],
        allow_delegation=False,
        verbose=True
    )
    
    return {
        'planner': planner_agent,
        'analyst': analyst_preprocessor_agent,
        'modeler': modeler_evaluator_agent
    }


def create_tasks(agents: Dict[str, Agent], executor_tool: CodeExecutor) -> List[Task]:
    """Create the three tasks for the pipeline."""
    
    # EXACT text from your notebook (only 'Units Sold' changed to 'Target')
    planning_task = Task(
        description=(
            "Create a comprehensive plan for regression analysis on the dataset in global DataFrame 'shared_df'. The target column to predict is named 'Target'.\n\n"
        "Your plan MUST follow the data science pipeline and include these phases:\n\n"
        "1. EXPLORATORY DATA ANALYSIS (EDA):\n"
        "   - Goal: Inspect the dataset thoroughly - print shape, column names, data types, missing value counts, and statistical summaries.\n"
        "   - Goal: Identify which columns are numeric, categorical, date-type, or identifier columns (IDs, names, codes).\n"
        "   - The agent must use the 'Notebook Code Executor' tool to run inspection code.\n\n"
        "2. PREPROCESSING:\n"
        "   - Goal: Based on EDA findings, handle missing values appropriately (drop or impute based on column type).\n"
        "   - Goal: Handle outliers in numeric columns using capping (clip to percentiles or IQR method).\n"
        "   - Goal: Handle date columns if present (convert to datetime, extract features if useful, then drop).\n"
        "   - Goal: Drop identifier columns that should not be features (IDs, names, product codes).\n"
        "   - Goal: One-Hot Encode categorical columns with low cardinality.\n"
        "   - Goal: Apply standardization on numeric columns if necessary based on data distribution.\n"
        "   - Goal: Create global variables X (features) and y (target), then create train/test split (X_train, X_test, y_train, y_test) with 80/20 ratio.\n"
        "   - The agent must explain WHY each preprocessing step was taken.\n"
        "   - The agent must use the 'Notebook Code Executor' tool to run preprocessing code.\n\n"
        "3. MODELING:\n"
        "   - Goal: Train exactly 5 different regression models: LinearRegression, DecisionTreeRegressor, RandomForestRegressor, GradientBoostingRegressor, Ridge.\n"
        "   - Goal: Use random_state=42 for reproducibility where applicable.\n"
        "   - The agent must use the 'Notebook Code Executor' tool to run modeling code.\n\n"
        "4. EVALUATION:\n"
        "   - Goal: Calculate MAE, MSE, RMSE, and R² for each of the 5 models.\n"
        "   - Goal: Create a comparison table showing all models and their metrics.\n"
        "   - Goal: Identify which model is best for each metric (lowest MAE, lowest RMSE, highest R²).\n"
        "   - Goal: Extract and print top 10 feature importances from the best model, make sure the name of the features are there and not F-1, F-2 etc.The name of the features from the csv file should be used.\n"
        "   - Goal: Provide a final recommendation with reasoning about which model to use.\n"
        "   - The agent must use the 'Notebook Code Executor' tool to run evaluation code.\n\n"
        "Output a numbered plan with clear goals for each phase that subsequent agents can follow."
        ),
        expected_output=(
            "A numbered plan outlining the data science pipeline goals for subsequent agents. "
        "The plan must cover EDA, Preprocessing (including outlier handling and standardization), Modeling (with 5 specific models named), and Evaluation phases. "
        "Each goal must remind agents to use the 'Notebook Code Executor' tool to write and execute Python code. "
        "The plan must specify that the target column is 'Target' and that preprocessing decisions should be based on actual data inspection."
        ),
        agent=agents['planner']
    )
    
    # EXACT text from your notebook (only 'Units Sold' changed to 'Target')
    data_analysis_preprocessing_task = Task(
        description=(
            "**CRITICAL: You MUST use the 'Notebook Code Executor' tool to execute code. Do NOT just write code - CALL the tool.**\n\n"
        "Follow the analysis and preprocessing plan. Your goal is to thoroughly inspect the global 'shared_df' DataFrame, perform appropriate preprocessing based on your findings, and create the global training/testing variables.\n\n"
        "You MUST generate Python code to achieve this and execute it using the 'Notebook Code Executor' tool.\n\n"
        "STEP 1 - EXPLORATORY DATA ANALYSIS:\n"
        "Write code to inspect the dataset:\n"
        "- Print the shape of shared_df using print(shared_df.shape)\n"
        "- Print column names and data types using print(shared_df.info())\n"
        "- Print missing value counts using print(shared_df.isnull().sum())\n"
        "- Print statistical summary using print(shared_df.describe())\n"
        "- Print the first few rows using print(shared_df.head())\n"
        "- Identify column types: numeric columns (int64, float64), categorical columns (object), potential date columns, potential identifier columns\n\n"
        "STEP 2 - PREPROCESSING (based on your EDA findings):\n"
        "For EACH preprocessing step, print a message explaining your reasoning.\n\n"
        "2a. Missing Values: If any columns have missing values, handle them appropriately:\n"
        "- For numeric columns: fill with median or mean\n"
        "- For categorical columns: fill with mode or 'Unknown'\n"
        "- Or drop rows if missing count is very small\n"
        "- IMPORTANT: Do NOT use inplace=True. Use assignment instead: shared_df[col] = shared_df[col].fillna(value)\n"
        "- Print: 'Handling missing values in [column] by [method] because [reason]'\n\n"
        "2b. Outliers: Handle outliers in numeric columns using capping:\n"
        "- Calculate the 1st percentile (lower bound) and 99th percentile (upper bound) for each numeric column\n"
        "- Clip values outside these bounds using df[column].clip(lower, upper)\n"
        "- Alternatively use IQR method: lower = Q1 - 1.5*IQR, upper = Q3 + 1.5*IQR\n"
        "- Print: 'Capping outliers in [column] to [lower, upper] range'\n\n"
        "2c. Date Columns: If you identify any date columns:\n"
        "- Convert to datetime using pd.to_datetime(col, format='mixed', dayfirst=True, errors='coerce'). This handles varied date formats and converts unparseable values to NaT instead of crashing.\n"
        "- After conversion, check for and report any NaT values created by failed parsing.\n"
        "- Optionally sort the DataFrame by date\n"
        "- Drop the date column (or extract useful features like year, month first)\n"
        "- Print: 'Dropping date column [name] because [reason]'\n\n"
        "2d. Identifier Columns: Drop columns that should not be features:\n"
        "- Columns containing IDs, names, codes, unique identifiers\n"
        "- Print: 'Dropping identifier column [name] because [reason]'\n\n"
        "2e. Categorical Encoding: For categorical columns (object dtype):\n"
        "- Apply One-Hot Encoding using pd.get_dummies(shared_df, columns=[list], drop_first=True)\n"
        "- Update shared_df with the encoded result\n"
        "- Print: 'One-Hot Encoding column [name] because [reason]'\n\n"
        "2f. Standardization: If numeric features have very different scales:\n"
        "- Apply StandardScaler from sklearn.preprocessing if you find it necessary\n"
        "- Print: 'Applying standardization because [reason]' or 'Skipping standardization because [reason]'\n\n"
        "STEP 3 - CREATE TRAIN/TEST SPLIT:\n"
        "- FIRST: Drop any remaining rows with NaN values using shared_df = shared_df.dropna(). Print how many rows were dropped.\n"
        "- Create global variable 'y' = shared_df['Target']\n"
        "- Create global variable 'X' = shared_df.drop('Target', axis=1)\n"
        "- Use train_test_split to create X_train, X_test, y_train, y_test with test_size=0.2, shuffle=True, random_state=42\n"
        "- Print shapes: X_train.shape, X_test.shape, y_train.shape, y_test.shape\n\n"
        "Make sure your code includes all necessary imports (pandas, numpy, train_test_split from sklearn.model_selection, StandardScaler from sklearn.preprocessing if used).\n\n"
        "IMPORTANT: You MUST call the 'Notebook Code Executor' tool to run your code. DO NOT just write code in your response - actually CALL the tool. Your task FAILS if you do not invoke the tool."
        ),
        expected_output=(
            "Output from the 'Notebook Code Executor' tool showing successful execution of the agent-generated code. This must include:\n"
        "- EDA results: shape, info, missing values, describe output\n"
        "- Preprocessing explanations: printed messages explaining WHY each preprocessing decision was made (missing values, outliers, date columns, identifiers, encoding, standardization)\n"
        "- Confirmation of train/test split creation with printed shapes of X_train, X_test, y_train, y_test\n"
        "- The agent must have actually CALLED the tool, not just written code in the response"
        ),
        agent=agents['analyst'],
        tools=[executor_tool]
    )
    
    # EXACT text from your notebook
    modeling_evaluation_task = Task(
        description=(
             "**CRITICAL: You MUST use the 'Notebook Code Executor' tool to execute code. Do NOT just write code - CALL the tool.**\n\n"
            "You MUST generate Python code assuming global variables X_train, X_test, y_train, y_test exist, and execute it using the 'Notebook Code Executor' tool.\n\n"
            "STEP 0 - DATA VALIDATION:\n"
            "Before training any models, check for and handle NaN values:\n"
            "- Print X_train.isnull().sum().sum() and X_test.isnull().sum().sum()\n"
            "- If any NaN values exist, drop those rows: X_train = X_train.dropna() and adjust y_train accordingly using the same index\n"
            "- Do the same for X_test and y_test\n"
            "- Print 'Data validated, no NaN values remain' after cleanup\n\n"
            "STEP 1 - TRAIN 5 REGRESSION MODELS:\n"
        "Train exactly these 5 models and store them in a dictionary:\n"
        "- LinearRegression() - import from sklearn.linear_model\n"
        "- DecisionTreeRegressor(random_state=42) - import from sklearn.tree\n"
        "- RandomForestRegressor(random_state=42) - import from sklearn.ensemble\n"
        "- GradientBoostingRegressor(random_state=42) - import from sklearn.ensemble\n"
        "- Ridge(random_state=42) - import from sklearn.linear_model\n\n"
        "For each model:\n"
        "- Call model.fit(X_train, y_train) to train\n"
        "- Print: 'Training [model_name]...'\n\n"
        "STEP 2 - EVALUATE ALL MODELS:\n"
        "For each of the 5 trained models:\n"
        "- Make predictions on X_test using model.predict(X_test)\n"
        "- Calculate these metrics by comparing predictions to y_test:\n"
        "  - MAE (Mean Absolute Error) using mean_absolute_error from sklearn.metrics\n"
        "  - MSE (Mean Squared Error) using mean_squared_error from sklearn.metrics\n"
        "  - RMSE (Root Mean Squared Error) = sqrt(MSE) using numpy\n"
        "  - R² (R-squared) using r2_score from sklearn.metrics\n"
        "- Store all metrics for comparison\n\n"
        "STEP 3 - CREATE COMPARISON TABLE:\n"
        "Create and print a pandas DataFrame showing all models and their metrics:\n"
        "- Columns: Model Name, MAE, MSE, RMSE, R²\n"
        "- One row per model\n"
        "- Print the full table\n\n"
        "STEP 4 - IDENTIFY BEST MODELS:\n"
        "Print which model is best for each metric:\n"
        "- 'Best MAE: [model_name] with MAE = [value]' (lowest MAE is best)\n"
        "- 'Best MSE: [model_name] with MSE = [value]' (lowest MSE is best)\n"
        "- 'Best RMSE: [model_name] with RMSE = [value]' (lowest RMSE is best)\n"
        "- 'Best R²: [model_name] with R² = [value]' (highest R² is best)\n\n"
        "STEP 5 - FEATURE IMPORTANCES:\n"
        "For the model with the best overall performance (consider R² as the primary metric):\n"
        "- Extract feature importances (use model.feature_importances_ for tree-based models)\n"
        "- If the best model doesn't have feature_importances_ (like LinearRegression), use RandomForest instead for feature importance\n"
        "- Create a DataFrame with columns: Feature, Importance\n"
        "- Sort by importance descending\n"
        "- Print the top 10 most important features,make sure the name of the features are there and not F-1, F-2 etc.The name of the features from the csv file should be used.\n\n"
        "STEP 6 - FINAL RECOMMENDATION:\n"
        "Print a recommendation section:\n"
        "- State which model you recommend overall\n"
        "- Explain WHY based on the metrics comparison\n"
        "- Note any trade-offs between models\n\n"
        "Make sure your code includes all necessary imports (sklearn models, metrics, numpy, pandas).\n\n"
        "IMPORTANT: You MUST call the 'Notebook Code Executor' tool to run your code. DO NOT just write code in your response - actually CALL the tool. Your task FAILS if you do not invoke the tool."
        ),
        expected_output=(
            "Output from the 'Notebook Code Executor' tool showing successful execution including:\n"
        "- Training confirmation for all 5 models\n"
        "- A comparison table showing all models with MAE, MSE, RMSE, R² metrics\n"
        "- Identification of which model is best for each metric\n"
        "- Top 10 feature importances with feature names and importance values, make sure the name of the features are there and not F-1, F-2 etc.The name of the features from the csv file should be used.\n"
        "- A final recommendation explaining which model to use and why\n"
        "- The agent must have actually CALLED the tool, not just written code in the response."
        ),
        agent=agents['modeler'],
        tools=[executor_tool]
    )
    
    return [planning_task, data_analysis_preprocessing_task, modeling_evaluation_task]


def run_pipeline(csv_path: str, job_id: str, jobs_dir: str = "jobs") -> Dict[str, Any]:
    """
    Main entry point - runs the CrewAI pipeline on any dataset.
    
    Args:
        csv_path: Path to the uploaded CSV file
        job_id: Unique identifier for this job
        jobs_dir: Directory where job outputs are stored
        
    Returns:
        Dictionary with status, job_id, output, logs, error
    """
    
    # Setup job directory
    job_path = Path(jobs_dir) / job_id
    job_path.mkdir(parents=True, exist_ok=True)
    
    # Create isolated namespace for this run
    namespace = {}
    
    try:
        # Load CSV into namespace as 'shared_df'
        df = pd.read_csv(csv_path)
        namespace['shared_df'] = df
        namespace['pd'] = pd
        namespace['np'] = np
        
        # Create code executor with this namespace
        executor_tool = CodeExecutor(namespace=namespace)
        
        # Create LLM
        llm = create_llm()
        
        # Create agents and tasks
        agents = create_agents(llm, executor_tool)
        tasks = create_tasks(agents, executor_tool)
        
        # Create and run crew
        crew = Crew(
            agents=[agents['planner'], agents['analyst'], agents['modeler']],
            tasks=tasks,
            process=Process.sequential,
            verbose=1,
            output_log_file=str(job_path / "crew_log.txt")
        )
        
        crew_result = crew.kickoff()
        
        # Get the crew's final output
        output = str(crew_result.raw) if hasattr(crew_result, 'raw') else str(crew_result)
        
        # Read logs
        log_file = job_path / "crew_log.txt"
        logs = ""
        if log_file.exists():
            logs = log_file.read_text(encoding='utf-8', errors='ignore')
        
        result = {
            "status": "success",
            "job_id": job_id,
            "output": output,
            "logs": logs,
            "error": None
        }
        
        with open(job_path / "results.json", 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        error_result = {
            "status": "error",
            "job_id": job_id,
            "output": None,
            "logs": "",
            "error": f"Pipeline error: {type(e).__name__}: {str(e)}"
        }
        
        with open(job_path / "results.json", 'w') as f:
            json.dump(error_result, f, indent=2)
        
        return error_result
