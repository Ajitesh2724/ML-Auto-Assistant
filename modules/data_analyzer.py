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