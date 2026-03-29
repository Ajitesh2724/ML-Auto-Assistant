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
        """
        task_type: 'classification' or 'regression'
        """
        self.task_type = task_type

    def variance_threshold(self, X, threshold=0.0):
        """
        Removes features with low variance
        """
        selector = VarianceThreshold(threshold=threshold)
        X_selected = selector.fit_transform(X)

        selected_features = X.columns[selector.get_support()]

        print(f"[FeatureSelector] Variance Threshold selected {len(selected_features)} features")
        return pd.DataFrame(X_selected, columns=selected_features)

    def select_k_best(self, X, y, k=10, method="f_score"):
        """
        Select top k features using statistical tests
        method:
            - 'f_score'
            - 'mutual_info'
        """

        if self.task_type == "classification":
            if method == "f_score":
                score_func = f_classif
            else:
                score_func = mutual_info_classif
        else:
            if method == "f_score":
                score_func = f_regression
            else:
                score_func = mutual_info_regression

        selector = SelectKBest(score_func=score_func, k=k)
        X_selected = selector.fit_transform(X, y)

        selected_features = X.columns[selector.get_support()]

        print(f"[FeatureSelector] SelectKBest picked {len(selected_features)} features")
        return pd.DataFrame(X_selected, columns=selected_features)

    def random_forest_importance(self, X, y, n_estimators=100, top_n=10):
        """
        Select features based on Random Forest importance
        """

        if self.task_type == "classification":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)

        model.fit(X, y)

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "feature": X.columns,
            "importance": importances
        }).sort_values(by="importance", ascending=False)

        selected_features = feature_importance_df.head(top_n)["feature"].values

        print(f"[FeatureSelector] RandomForest selected top {top_n} features")
        return X[selected_features]

    def correlation_filter(self, X, threshold=0.9):
        """
        Removes highly correlated features
        """
        corr_matrix = X.corr().abs()

        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]

        print(f"[FeatureSelector] Dropping {len(to_drop)} highly correlated features")
        return X.drop(columns=to_drop)

    def auto_select(self, X, y):
        """
        Combined pipeline:
        1. Remove correlated features
        2. Select important features using RF
        """

        print("[FeatureSelector] Starting automatic feature selection...")

        X = self.correlation_filter(X)
        X = self.random_forest_importance(X, y, top_n=min(10, X.shape[1]))

        print("[FeatureSelector] Feature selection complete")
        return X