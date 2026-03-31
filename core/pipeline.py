import pandas as pd
from sklearn.model_selection import train_test_split

from modules.data_profiler import DataProfiler
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector
from modules.model_trainer import ModelTrainer
from modules.model_evaluator import evaluate_model


# -------------------------------
# TASK DETECTION
# -------------------------------
def detect_task_type(y):

    if y.dtype == "object":
        return "classification"

    if y.nunique() <= 20:
        return "classification"

    return "regression"


# -------------------------------
# FEATURE IMPORTANCE EXTRACTOR
# -------------------------------
def get_feature_importance(model, feature_names):

    importance = None

    # tree based models
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    # linear models
    elif hasattr(model, "coef_"):

        importance = model.coef_

        # multi-class case
        if hasattr(importance, "shape") and len(importance.shape) > 1:
            importance = importance[0]

        importance = abs(importance)

    # model does not support importance
    if importance is None:
        return None

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    )

    return importance_df


# -------------------------------
# MAIN PIPELINE
# -------------------------------
def run_pipeline(
    df,
    target_column,
    task_type="auto"
):

    try:

        # -------------------------------
        # TASK TYPE
        # -------------------------------
        if task_type == "auto":
            task_type = detect_task_type(df[target_column])

        # -------------------------------
        # PROFILING
        # -------------------------------
        profiler = DataProfiler(df)
        numeric_cols, categorical_cols, _ = profiler.summary()

        # -------------------------------
        # HANDLE MISSING
        # -------------------------------
        df = df.fillna(0)

        # -------------------------------
        # ENCODING
        # -------------------------------
        df = encode_categorical(
            df,
            method="onehot",
            target_column=target_column
        )

        # -------------------------------
        # SCALING
        # -------------------------------
        df = scale_features(
            df,
            target_column=target_column
        )

        # -------------------------------
        # SPLIT
        # -------------------------------
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        # -------------------------------
        # FEATURE SELECTION
        # -------------------------------
        selector = FeatureSelector(task_type=task_type)

        X_train = selector.auto_select(
            X_train,
            y_train
        )

        # align test columns safely
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

        # -------------------------------
        # MODEL TRAINING
        # -------------------------------
        trainer = ModelTrainer(task_type=task_type)

        trainer.train(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test
        )

        preds = trainer.predict(X_test)

        # -------------------------------
        # EVALUATION
        # -------------------------------
        metrics = evaluate_model(
            y_test,
            preds,
            problem_type=task_type
        )

        # -------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------
        importance_df = get_feature_importance(
            trainer.best_model,
            X_train.columns
        )

        if importance_df is not None:
            importance_dict = importance_df.to_dict()
        else:
            importance_dict = None

        # -------------------------------
        # RETURN
        # -------------------------------
        return {

            "success": True,

            "task_type": task_type,

            "model": trainer.best_model_name,

            "metrics": metrics,

            "features": list(X_train.columns),

            "feature_importance": importance_dict
        }

    except Exception as e:

        return {

            "success": False,

            "error": str(e)
        }