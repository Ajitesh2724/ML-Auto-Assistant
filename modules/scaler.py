from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd


def scale_features(df, method="standard", target_column=None):
    """
    Robust feature scaling for real datasets
    """

    df = df.copy()

    # numeric columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()

    # remove target column
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    # remove binary columns (0/1)
    binary_cols = [col for col in numeric_cols if df[col].nunique() <= 2]
    numeric_cols = [col for col in numeric_cols if col not in binary_cols]

    if len(numeric_cols) == 0:
        print("⚠️ No numeric columns to scale")
        return df

    # choose scaler
    if method == "standard":
        scaler = StandardScaler()

    elif method == "minmax":
        scaler = MinMaxScaler()

    elif method == "robust":
        scaler = RobustScaler()

    else:
        raise ValueError("Invalid scaling method")

    # scale
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # round (optional)
    df[numeric_cols] = df[numeric_cols].round(3)

    print("\nScaling done using:", method)
    print("Scaled columns:", numeric_cols)

    return df