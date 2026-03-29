import pandas as pd

def handle_missing_values(df, method="mean"):

    numeric_cols = df.select_dtypes(include=['int64','float64']).columns
    cat_cols = df.select_dtypes(include=['object']).columns

    if method == "mean":

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

    elif method == "median":

        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    elif method == "constant":

        df[numeric_cols] = df[numeric_cols].fillna(0)

        df[cat_cols] = df[cat_cols].fillna("Unknown")

    elif method == "ffill":

        df = df.fillna(method="ffill")

    elif method == "bfill":

        df = df.fillna(method="bfill")

    else:

        print("Invalid method selected")

    print("\nMissing values handled using:", method)

    return df