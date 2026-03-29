from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


def evaluate_model(y_true, y_pred, problem_type="classification"):

    results = {}

    if problem_type == "classification":

        results["accuracy"] = accuracy_score(y_true, y_pred)

        results["precision"] = precision_score(
            y_true,
            y_pred,
            average="weighted"
        )

        results["recall"] = recall_score(
            y_true,
            y_pred,
            average="weighted"
        )

        results["f1_score"] = f1_score(
            y_true,
            y_pred,
            average="weighted"
        )

    elif problem_type == "regression":

        results["MAE"] = mean_absolute_error(y_true, y_pred)

        results["MSE"] = mean_squared_error(y_true, y_pred)

        results["R2_score"] = r2_score(y_true, y_pred)

    else:

        print("Invalid problem type")

    print("\nModel Evaluation Results:")

    for metric, value in results.items():

        print(f"{metric}: {round(value,4)}")

    return results