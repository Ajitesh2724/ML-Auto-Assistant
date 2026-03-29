import pandas as pd

def handle_missing_values(df):

    numeric_columns = df.select_dtypes(include=['float64','int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Fill numeric columns with mean
    for col in numeric_columns:
        df[col] = df[col].fillna(df[col].mean())

    # Fill categorical columns with mode
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])

    print("\nMissing values handled successfully!")

    return df