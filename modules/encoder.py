import pandas as pd

def encode_categorical(df):

    categorical_columns = df.select_dtypes(include=['object']).columns

    df = pd.get_dummies(df, columns=categorical_columns)

    print("\nCategorical variables encoded successfully!")

    return df