import pandas as pd
from sklearn.model_selection import train_test_split

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
    try:
        logs = []

        # -------------------------------
        # TASK DETECTION
        # -------------------------------
        if task_type == "auto":
            task_type = detect_task_type(df[target_column])

        logs.append(f"Task detected: {task_type}")

        # -------------------------------
        # PROFILING
        # -------------------------------
        profiler = DataProfiler(df)
        numeric_cols, categorical_cols, _ = profiler.summary()

        # -------------------------------
        # HANDLE MISSING (SAFE)
        # -------------------------------
        df = df.fillna(0)
        logs.append("Missing values handled (fillna)")

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

        if len(X) == 0:
            raise ValueError("Dataset became empty after preprocessing")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # -------------------------------
        # FEATURE SELECTION
        # -------------------------------
        selector = FeatureSelector(task_type=task_type)
        X_train = selector.auto_select(X_train, y_train)

        if X_train.shape[1] == 0:
            raise ValueError("No features left after feature selection")

        # ✅ FIXED alignment
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

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
            "success": True,
            "task_type": task_type,
            "model": trainer.best_model_name,
            "metrics": results,
            "features": list(X_train.columns),
            "logs": logs
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }