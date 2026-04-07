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



def get_feature_importance(model, feature_names):

    importance = None

    # tree based models
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_

    
    elif hasattr(model, "coef_"):

        importance = model.coef_

        # multi-class case
        if hasattr(importance, "shape") and len(importance.shape) > 1:
            importance = importance[0]

        importance = abs(importance)

   
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



def run_pipeline(
    df,
    target_column,
    task_type="auto"
):

    logs = []

    def log(msg):
        print(msg)
        logs.append(msg)

    try:

        log("🚀 Pipeline started")

        if task_type == "auto":
            task_type = detect_task_type(df[target_column])

        log(f"[Task] Detected task type: {task_type}")

       
        profiler = DataProfiler(df)
        numeric_cols, categorical_cols, _ = profiler.summary()

        log(f"[DataProfiler] Numeric Columns: {numeric_cols}")
        log(f"[DataProfiler] Categorical Columns: {categorical_cols}")

       
        df = df.fillna(0)
        log("[Preprocessing] Missing values filled with 0")

       
        df = encode_categorical(
            df,
            method="onehot",
            target_column=target_column
        )
        log("[Preprocessing] Encoding applied: onehot")

        
        df = scale_features(
            df,
            target_column=target_column
        )
        log("[Preprocessing] Scaling applied: standard")

       
        X = df.drop(columns=[target_column])
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        log(f"[Split] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

  
        log("[FeatureSelector] Starting feature selection")

        selector = FeatureSelector(task_type=task_type)

        X_train = selector.auto_select(
            X_train,
            y_train
        )

        log(f"[FeatureSelector] Selected {len(X_train.columns)} features")

        # align test columns safely
        X_test = X_test.reindex(
            columns=X_train.columns,
            fill_value=0
        )

       
        log("[ModelTrainer] Training models...")

        trainer = ModelTrainer(task_type=task_type)

        trainer.train(
            X_train,
            y_train,
            X_val=X_test,
            y_val=y_test
        )

        log(f"[ModelTrainer] Best model: {trainer.best_model_name}")

        preds = trainer.predict(X_test)

        
        metrics = evaluate_model(
            y_test,
            preds,
            problem_type=task_type
        )

        log(f"[Evaluation] Metrics: {metrics}")

        # -------------------------------
        # FEATURE IMPORTANCE
        # -------------------------------
        importance_df = get_feature_importance(
            trainer.best_model,
            X_train.columns
        )

        if importance_df is not None:
            importance_dict = importance_df.to_dict()
            log("[FeatureImportance] Extracted successfully")
        else:
            importance_dict = None
            log("[FeatureImportance] Not available for this model")

        log("✅ Pipeline completed successfully")

        # -------------------------------
        # RETURN
        # -------------------------------
        return {

            "success": True,

            "task_type": task_type,

            "model": trainer.best_model_name,

            "metrics": metrics,

            "features": list(X_train.columns),

            "feature_importance": importance_dict,

            "logs": logs 
        }

    except Exception as e:

        log(f"[ERROR] {str(e)}")

        return {

            "success": False,

            "error": str(e),

            "logs": logs   # even errors will show logs
        }