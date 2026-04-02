from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.metrics import accuracy_score, r2_score

import joblib


class ModelTrainer:

    def __init__(
        self,
        task_type="classification",
        model_name="auto",
        random_state=42
    ):
        self.task_type = task_type
        self.model_name = model_name
        self.random_state = random_state

        self.best_model = None
        self.best_model_name = None


    
    def _get_models(self, n_samples):

        
        if self.task_type == "classification":

            models = {
                "logistic": LogisticRegression(max_iter=1000),
                "random_forest": RandomForestClassifier(
                    n_estimators=120,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }

            return models

 
        elif self.task_type == "regression":

            models = {
                "linear": LinearRegression(),
                "ridge": Ridge(),
                "random_forest": RandomForestRegressor(
                    n_estimators=120,
                    random_state=self.random_state,
                    n_jobs=-1
                )
            }

            return models

        else:
            raise ValueError("Invalid task type")



    def train(self, X_train, y_train, X_val=None, y_val=None):

        n_samples = X_train.shape[0]
        models = self._get_models(n_samples)

    
        if self.model_name != "auto":

            if self.model_name not in models:
                raise ValueError("Invalid model name")

            self.best_model = models[self.model_name]
            self.best_model.fit(X_train, y_train)

            self.best_model_name = self.model_name

            print(f"\nSelected Model: {self.best_model_name}")
            return self.best_model

        
        print("\nRunning Auto Model Selection...\n")

        best_score = -999

        for name, model in models.items():

            print(f"Training {name}...")

            try:
                model.fit(X_train, y_train)
                preds = model.predict(X_val)

                if self.task_type == "classification":
                    score = accuracy_score(y_val, preds)
                else:
                    score = r2_score(y_val, preds)

                print(f"{name} score: {score:.4f}")

                if score > best_score:
                    best_score = score
                    self.best_model = model
                    self.best_model_name = name

            except Exception as e:
                print(f"⚠️ Skipping {name} due to error: {e}")
                continue

        print("\nBest Model Selected:", self.best_model_name)
        print("Best Score:", round(best_score, 4))

        return self.best_model



    def predict(self, X_test):
        return self.best_model.predict(X_test)


    def save_model(self, path="best_model.pkl"):
        joblib.dump(self.best_model, path)
        print("\nModel saved at", path)