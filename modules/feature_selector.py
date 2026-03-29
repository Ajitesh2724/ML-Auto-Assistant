import pandas as pd
import numpy as np
from sklearn.feature_selection import (
    VarianceThreshold,
    SelectKBest,
    f_classif,
    f_regression,
    mutual_info_classif,
    mutual_info_regression
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


class FeatureSelector:
    def __init__(self, task_type="classification"):
        self.task_type = task_type

    # -------------------------------
    # DATA PREPARATION (SAFE)
    # -------------------------------
    def _prepare_data(self, X):
        X = X.copy()

        # Convert bool → int
        bool_cols = X.select_dtypes(include="bool").columns
        X[bool_cols] = X[bool_cols].astype(int)

        # Keep only numeric
        X = X.select_dtypes(include=[np.number])

        # Fill NaNs
        X = X.fillna(0)

        return X

    # -------------------------------
    # CORRELATION FILTER
    # -------------------------------
    def correlation_filter(self, X, threshold=0.95):
        X = self._prepare_data(X)

        corr_matrix = X.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        # Prevent over-dropping
        if len(to_drop) > 0.5 * X.shape[1]:
            print("[FeatureSelector] Too many correlated features → skipping filter")
            return X

        print(f"[FeatureSelector] Dropping {len(to_drop)} correlated features")
        return X.drop(columns=to_drop)

    # -------------------------------
    # FAST FEATURE SELECTION
    # -------------------------------
    def select_k_best(self, X, y, k=10, method="f_score"):
        X = self._prepare_data(X)

        if self.task_type == "classification":
            score_func = f_classif if method == "f_score" else mutual_info_classif
        else:
            score_func = f_regression if method == "f_score" else mutual_info_regression

        k = min(k, X.shape[1])

        print("[FeatureSelector] Running SelectKBest...")
        selector = SelectKBest(score_func=score_func, k=k)

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()]

        print(f"[FeatureSelector] SelectKBest selected {len(selected_features)} features")

        return pd.DataFrame(X_selected, columns=selected_features)

    # -------------------------------
    # RANDOM FOREST IMPORTANCE (OPTIMIZED)
    # -------------------------------
    def random_forest_importance(self, X, y, top_n=10):
        X = self._prepare_data(X)

        print("[FeatureSelector] Training RandomForest...")

        if self.task_type == "classification":
            model = RandomForestClassifier(
                n_estimators=30,
                max_depth=5,
                n_jobs=-1,
                random_state=42
            )
        else:
            model = RandomForestRegressor(
                n_estimators=30,
                max_depth=5,
                n_jobs=-1,
                random_state=42
            )

        model.fit(X, y)

        print("[FeatureSelector] RandomForest training complete")

        importances = model.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        top_n = min(top_n, len(feature_importance_df))
        selected_features = feature_importance_df.head(top_n)["feature"].values

        print(f"[FeatureSelector] RF selected top {top_n} features")

        return X[selected_features]

    # -------------------------------
    # VARIANCE FILTER (FALLBACK)
    # -------------------------------
    def variance_threshold(self, X, threshold=0.0):
        X = self._prepare_data(X)

        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)

        selected_features = X.columns[selector.get_support()]

        print(f"[FeatureSelector] Variance selected {len(selected_features)} features")

        return pd.DataFrame(X_selected, columns=selected_features)

    # -------------------------------
    # AUTO FEATURE SELECTION PIPELINE
    # -------------------------------
    def auto_select(self, X, y):
        print("\n[FeatureSelector] Starting automatic feature selection...")

        try:
            # Step 1: Correlation filter
            X = self.correlation_filter(X)

            # Step 2: Fast selection first
            X = self.select_k_best(
                X,
                y,
                k=min(15, X.shape[1])
            )

            # Step 3: Refine using RandomForest
            X = self.random_forest_importance(
                X,
                y,
                top_n=min(10, max(3, X.shape[1]))
            )

        except Exception as e:
            print(f"[FeatureSelector] Error: {e}")
            print("[FeatureSelector] Falling back to variance threshold")
            X = self.variance_threshold(X)

        print("[FeatureSelector] Feature selection complete\n")

        return X