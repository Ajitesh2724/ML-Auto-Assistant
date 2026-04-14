def generate_advanced_report(results, df=None):

    report = []

    report.append("ML AUTO ASSISTANT - ADVANCED REPORT")
    report.append("=" * 50)

    # Model Info
    report.append(f"\nModel: {results['model']}")
    report.append(f"Task Type: {results['task_type']}")

    # Metrics
    report.append("\n--- Metrics ---")
    for k, v in results["metrics"].items():
        report.append(f"{k}: {round(v, 4)}")

    # Features
    report.append("\n--- Selected Features ---")
    report.append(", ".join(results["features"]))

    # Feature Importance
    if results.get("feature_importance"):
        report.append("\n--- Feature Importance ---")
        fi = results["feature_importance"]
        for f, imp in zip(fi["feature"], fi["importance"]):
            report.append(f"{f}: {round(imp, 4)}")

    # Logs
    report.append("\n--- Pipeline Logs ---")
    for log in results.get("logs", []):
        report.append(log)

    return "\n".join(report)