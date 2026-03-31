import pandas as pd
from sklearn.model_selection import train_test_split

from modules.data_analyzer import analyze_data
from modules.data_profiler import DataProfiler
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import evaluate_model


def detect_task_type(y):
    if y.dtype == "object":
        return "classification"
    if y.nunique() <= 20:
        return "classification"
    return "regression"


def run_pipeline(
    df,
    target_column,
    task_type="auto",
    use_llm=False
):
    """
    Main pipeline wrapper for UI
    """

    logs = []

    # -------------------------------
    # TASK DETECTION
    # -------------------------------
    if task_type == "auto":
        task_type = detect_task_type(df[target_column])

    logs.append(f"Task detected: {task_type}")

    # -------------------------------
    # ANALYSIS
    # -------------------------------
    analyze_data(df)  # prints (optional)

    profiler = DataProfiler(df)
    numeric_cols, categorical_cols, _ = profiler.summary()

    # -------------------------------
    # HANDLE MISSING
    # -------------------------------
    df = df.dropna()
    logs.append("Missing values handled (dropna)")

    # -------------------------------
    # ENCODING
    # -------------------------------
    df = encode_categorical(df, method="onehot", target_column=target_column)

    # -------------------------------
    # SCALING
    # -------------------------------
    df = scale_features(df, target_column=target_column)

    # -------------------------------
    # SPLIT
    # -------------------------------
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # -------------------------------
    # FEATURE SELECTION
    # -------------------------------
    selector = FeatureSelector(task_type=task_type)
    X_train = selector.auto_select(X_train, y_train)

    # align test set
    X_test = X_test[X_train.columns]

    # -------------------------------
    # MODEL TRAINING
    # -------------------------------
    trainer = ModelTrainer(task_type=task_type)

    model = trainer.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test
    )

    # -------------------------------
    # PREDICTION
    # -------------------------------
    preds = trainer.predict(X_test)

    # -------------------------------
    # EVALUATION
    # -------------------------------
    results = evaluate_model(
        y_test,
        preds,
        problem_type=task_type
    )

    return {
        "task_type": task_type,
        "model": trainer.best_model_name,
        "metrics": results,
        "features": list(X_train.columns),
        "logs": logs
    }