from modules import (
    data_analyzer,
    data_profiler,
    encoder,
    feature_selector,
    missing_handler,
    model_trainer,
    model_evaluator,
    scaler
)

TOOLS = {

    "analyze":
        lambda ctx, **k:
            data_analyzer.analyze_data(ctx["df"]),

    "profile":
        lambda ctx, **k:
            data_profiler.profile_data(ctx["df"]),

    "handle_missing":
        lambda ctx, **k:
            missing_handler.handle_missing_values(ctx["df"]),

    "encode":
        lambda ctx, **k:
            encoder.encode_data(ctx["df"]),

    "scale":
        lambda ctx, **k:
            scaler.scale_features(ctx["df"]),

    "feature_select":
        lambda ctx, **k:
            feature_selector.select_features(ctx["df"]),

    "train_model":
        lambda ctx, model="auto", **k:
            model_trainer.train_model(ctx["df"], model=model),

    "evaluate":
        lambda ctx, **k:
            model_evaluator.evaluate_model(ctx["df"]),
}