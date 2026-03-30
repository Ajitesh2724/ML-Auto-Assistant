import pandas as pd

from llm_layer.llm_agent import LLMAgent
from llm_layer.decision_engine import DecisionEngine

# existing modules (unchanged)
from modules.data_analyzer import analyze_data
from modules.data_profiler import DataProfiler
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import evaluate_model

from sklearn.model_selection import train_test_split


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(
    r"C:\Users\ajite\OneDrive\Desktop\ML-Auto-Assistant\data\sleep_health_dataset.csv"
)

print("Dataset Loaded Successfully\n")


# -------------------------------
# TARGET SELECTION (same as before)
# -------------------------------
print("Available columns:", df.columns.tolist())
target_column = input("Enter target column: ").strip()

if target_column not in df.columns:
    raise ValueError("Invalid target column selected!")


# -------------------------------
# TASK TYPE DETECTION (reuse your function)
# -------------------------------
def detect_task_type(y):

    if y.dtype == "object":
        return "classification"

    if y.nunique() <= 20:
        return "classification"

    return "regression"


task_type = detect_task_type(df[target_column])

print("\nDetected task:", task_type)


# -------------------------------
# CREATE CONTEXT FOR LLM
# -------------------------------
context = {

    "df": df,

    "target": target_column,

    "task_type": task_type,

    "steps_done": [],

    "missing":

        df.isnull().sum().sum() > 0,

    "categorical":

        len(df.select_dtypes(include="object").columns) > 0,

    "scaled": False,

    "trained": False

}


# -------------------------------
# INITIALIZE LLM
# -------------------------------
agent = LLMAgent()

engine = DecisionEngine()


# -------------------------------
# LLM LOOP
# -------------------------------
for step in range(10):

    print("\n====================")
    print("STEP", step)

    decision = agent.decide(context)

    print("LLM decision:", decision)


    result = engine.run(decision, context)

    if isinstance(result, dict) and result.get("stop"):

        break


# -------------------------------
# OPTIONAL: SHOW FINAL DATA
# -------------------------------
print("\nPipeline completed")