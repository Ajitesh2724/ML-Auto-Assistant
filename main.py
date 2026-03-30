
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
# TASK TYPE DETECTION (ROBUST)
# -------------------------------
def detect_task_type(y):
    import pandas as pd

    # Object or categorical → classification
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"

    # Few unique values → classification
    if y.nunique() <= 20:
        return "classification"

    return "regression"


# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(
    r"C:\Users\abhra\OneDrive\Documents\GitHub\ML-Auto-Assistant\data\sleep_health_dataset.csv"
)

print("Dataset Loaded Successfully\n")


# -------------------------------
# ANALYZE DATA
# -------------------------------
analyze_data(df)


# -------------------------------
# DATA PROFILING
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
# TARGET SELECTION
# -------------------------------
print("\nAvailable columns:", df.columns.tolist())
target_column = input("Enter target column: ").strip()

if target_column not in df.columns:
    raise ValueError("Invalid target column selected!")


# -------------------------------
# SPLIT X, y (BEFORE ENCODING)
# -------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]


# -------------------------------
# DETECT TASK TYPE (FIXED)
# -------------------------------
task_type = detect_task_type(y)

print(f"\nDetected Task Type: {task_type}")
print("Target dtype:", y.dtype)
print("Unique values:", y.nunique())


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
# SPLIT AGAIN AFTER PROCESSING
# -------------------------------
X = df.drop(columns=[target_column])
y = df[target_column]


# -------------------------------
# SAFETY CHECK (CRITICAL)
# -------------------------------
if task_type == "regression" and y.nunique() <= 20:
    raise ValueError(
        "❌ Regression selected but target looks categorical. Check detection logic!"
    )


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
# FINAL TARGET SAFETY (PREVENT YOUR ERROR)
# -------------------------------
if task_type == "classification":
    from sklearn.preprocessing import LabelEncoder

    if pd.api.types.is_object_dtype(y_train) or str(y_train.dtype) == "category":
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)


# -------------------------------
# MODEL TRAINING
# -------------------------------
trainer = ModelTrainer(
    task_type=task_type,
    model_name="random_forest"
)

model = trainer.train(X_train, y_train)


# -------------------------------
# PREDICTION
# -------------------------------
y_pred = trainer.predict(X_test)


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