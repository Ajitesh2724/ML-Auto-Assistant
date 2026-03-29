import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import evaluate_model
from sklearn.model_selection import train_test_split



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
df = pd.read_csv(r"C:\Users\ajite\OneDrive\Desktop\ML-Auto-Assistant\data\sleep_health_dataset.csv")

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

        model_name="random_forest"   # can change later

)

model = trainer.train(

        X_train,

        y_train

)


# -------------------------------
# MODEL PREDICTION
# -------------------------------

y_pred = trainer.predict(

        X_test

)


# -------------------------------
# MODEL EVALUATION
# -------------------------------

results = evaluate_model(

        y_test,

        y_pred,

        problem_type=task_type

)

print("\nFinal Evaluation Results:")

print(results)