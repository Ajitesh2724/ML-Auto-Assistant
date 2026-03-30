# llm_layer/tool_registry.py

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
    "eda": data_analyzer.analyze_data,
    "profile": data_profiler.profile_data,
    "handle_missing": missing_handler.handle_missing_values,
    "encode": encoder.encode_data,
    "scale": scaler.scale_features,
    "feature_select": feature_selector.select_features,
    "train_model": model_trainer.train_model,
    "evaluate": model_evaluator.evaluate_model,
}