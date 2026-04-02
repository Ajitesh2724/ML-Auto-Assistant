from modules import (
    data_analyzer,
    encoder,
    feature_selector,
    missing_handler,
    model_trainer,
    model_evaluator,
    scaler
)

from sklearn.model_selection import train_test_split


def analyze_tool(ctx):
    data_analyzer.analyze_data(ctx["df"])
    return ctx



def handle_missing_tool(ctx):
    ctx["df"] = missing_handler.handle_missing_values(ctx["df"])
    ctx["missing"] = False
    return ctx



def encode_tool(ctx):
    ctx["df"] = encoder.encode_categorical(
        ctx["df"],
        method="onehot",
        target_column=ctx["target"]
    )
    ctx["categorical"] = False
    return ctx


def scale_tool(ctx):
    ctx["df"] = scaler.scale_features(
        ctx["df"],
        target_column=ctx["target"]
    )
    ctx["scaled"] = True
    return ctx


def feature_select_tool(ctx):
    fs = feature_selector.FeatureSelector(
        task_type=ctx["task_type"]
    )

    X = ctx["df"].drop(columns=[ctx["target"]])
    y = ctx["df"][ctx["target"]]

    X_selected = fs.auto_select(X, y)

    # Rebuild dataset
    ctx["df"] = X_selected.copy()
    ctx["df"][ctx["target"]] = y.reset_index(drop=True)

    return ctx



def train_model_tool(ctx):
    X = ctx["df"].drop(columns=[ctx["target"]])
    y = ctx["df"][ctx["target"]]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trainer = model_trainer.ModelTrainer(
        task_type=ctx["task_type"]
    )

    model = trainer.train(X_train, y_train, X_val, y_val)

    ctx["model"] = model
    ctx["trained"] = True

    return ctx



def evaluate_tool(ctx):
    X = ctx["df"].drop(columns=[ctx["target"]])
    y = ctx["df"][ctx["target"]]

    preds = ctx["model"].predict(X)

    model_evaluator.evaluate_model(
        y,
        preds,
        ctx["task_type"]
    )

    ctx["evaluated"] = True
    return ctx


TOOLS = {
    "analyze": analyze_tool,
    "handle_missing": handle_missing_tool,
    "encode": encode_tool,
    "scale": scale_tool,
    "feature_select": feature_select_tool,
    "train_model": train_model_tool,
    "evaluate": evaluate_tool,
}