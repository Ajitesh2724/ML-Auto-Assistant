import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector


def detect_task_type(y):
    if y.dtype == "object":
        return "classification"

    if y.dtype in ["int64", "float64"]:
        # continuous float → regression
        if y.dtype == "float64":
            return "regression"

        # integer → check unique values
        if y.nunique() < 10:
            return "classification"
        else:
            return "regression"

    return "regression"


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("data/sample.csv")

print("Dataset Loaded Successfully\n")

# -------------------------------
# ANALYZE DATA
# -------------------------------
analyze_data(df)

# handle missing values
df = handle_missing_values(df)

# encode categorical variables
df = encode_categorical(df)

# scale features
df = scale_features(df)

print("\nFinal Preprocessed Dataset:\n")
print(df.head())

# -------------------------------
# TARGET SELECTION
# -------------------------------
print("\nAvailable columns:", df.columns.tolist())

target_column = input("Enter target column: ")

if target_column not in df.columns:
    raise ValueError("Invalid target column selected!")

# Split dataset
X = df.drop(columns=[target_column])
y = df[target_column]

# -------------------------------
# TASK TYPE DETECTION (FIX)
# -------------------------------
task_type = detect_task_type(y)

print(f"\nDetected Task Type: {task_type}")
print("Target dtype:", y.dtype)
print("Unique values:", y.nunique())

# -------------------------------
# FEATURE SELECTION
# -------------------------------
selector = FeatureSelector(task_type=task_type)

X_selected = selector.auto_select(X, y)

print("\nSelected Features:\n")
print(X_selected.head())