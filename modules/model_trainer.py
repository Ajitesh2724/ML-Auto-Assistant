from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR


class ModelTrainer:

    def __init__(self, task_type="classification", model_name="random_forest"):

        self.task_type = task_type

        self.model_name = model_name

        self.model = self._select_model()


    def _select_model(self):

        if self.task_type == "classification":

            models = {

                "logistic": LogisticRegression(),

                "random_forest": RandomForestClassifier(),

                "svm": SVC()

            }

        elif self.task_type == "regression":

            models = {

                "linear": LinearRegression(),

                "random_forest": RandomForestRegressor(),

                "svr": SVR()

            }

        else:

            raise ValueError("Invalid task type")


        if self.model_name not in models:

            raise ValueError("Invalid model name")


        print(f"\nSelected Model: {self.model_name}")


        return models[self.model_name]


    def train(self, X_train, y_train):

        self.model.fit(X_train, y_train)

        print("\nModel training completed")

        return self.model


    def predict(self, X_test):

        predictions = self.model.predict(X_test)

        print("\nPrediction completed")

        return predictions