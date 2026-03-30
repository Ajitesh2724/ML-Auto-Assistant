import pandas as pd

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
# TASK TYPE DETECTION
# -------------------------------
def detect_task_type(y):
    if y.dtype == "object":
        return "classification"

    if y.dtype == "float64":
        return "regression"

    if y.dtype == "int64":
        if y.nunique() < 10:
            return "classification"
        return "regression"

    return "regression"


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(
    r"C:\Users\ajite\OneDrive\Desktop\ML-Auto-Assistant\data\sleep_health_dataset.csv"
)

print("Dataset Loaded Successfully\n")


# -------------------------------
# ANALYZE DATA
# -------------------------------
analyze_data(df)


# -------------------------------
# DATA PROFILING (NEW)
# -------------------------------
profiler = DataProfiler(df)
numeric_cols, categorical_cols, id_cols = profiler.summary()


# -------------------------------
# DROP ID COLUMNS
# -------------------------------
df = df.drop(columns=id_cols, errors="ignore")


# -------------------------------
# HANDLE MISSING VALUES
# -------------------------------
df = handle_missing_values(df)


# -------------------------------
# TARGET SELECTION (EARLY)
# -------------------------------
print("\nAvailable columns:", df.columns.tolist())
target_column = input("Enter target column: ").strip()

if target_column not in df.columns:
    raise ValueError("Invalid target column selected!")


# -------------------------------
# ENCODING (SAFE)
# -------------------------------
df = encode_categorical(
    df,
    method="onehot",
    target_column=target_column
)


# -------------------------------
# CONVERT BOOL → INT
# -------------------------------
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)


# -------------------------------
# SCALING (SAFE)
# -------------------------------
df = scale_features(
    df,
    method="standard",
    target_column=target_column
)


print("\nFinal Preprocessed Dataset:\n")
print(df.head())


# -------------------------------
# SPLIT X, y
# -------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]


# -------------------------------
# DETECT TASK TYPE
# -------------------------------
task_type = detect_task_type(y)

print(f"\nDetected Task Type: {task_type}")
print("Target dtype:", y.dtype)
print("Unique values:", y.nunique())


# -------------------------------
# FEATURE SELECTION
# -------------------------------
selector = FeatureSelector(task_type=task_type)

if X.shape[1] > 1:
    X_selected = selector.auto_select(X, y)
else:
    X_selected = X

print("\nSelected Features:\n")
print(X_selected.head())


# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_selected,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTrain shape:", X_train.shape)
print("Test shape:", X_test.shape)


# -------------------------------
# MODEL TRAINING
# -------------------------------
trainer = ModelTrainer(
    task_type=task_type,
    model_name="auto"
)

model = trainer.train(
    X_train,
    y_train,
    X_test,
    y_test
)


# -------------------------------
# PREDICTION
# -------------------------------
y_pred = trainer.predict(X_test)

print(y_pred)
# -------------------------------
# EVALUATION
# -------------------------------
results = evaluate_model(
    y_test,
    y_pred,
    problem_type=task_type
)

print("\nFinal Evaluation Results:")
print(results)