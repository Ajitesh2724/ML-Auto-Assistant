import pandas as pd

def analyze_data(df):

    print("\nDataset Shape:")
    print(df.shape)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nData Types:")
    print(df.dtypes)

    # NEW ADDITIONS

    print("\nSummary Statistics:")
    print(df.describe())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nCorrelation Matrix:")
    print(df.corr(numeric_only=True))

    # Outlier detection using IQR
    numeric_cols = df.select_dtypes(include=['int64','float64']).columns

    print("\nOutlier Detection (IQR method):")

    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)

        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        outliers = df[(df[col] < lower) | (df[col] > upper)]

        print(f"{col}: {len(outliers)} outliers")