import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

def encode_categorical(df, method="onehot", target_column=None):

    cat_cols = df.select_dtypes(include=['object']).columns

    if method == "onehot":

        df = pd.get_dummies(df, columns=cat_cols)

    elif method == "label":

        le = LabelEncoder()

        for col in cat_cols:

            df[col] = le.fit_transform(df[col])

    elif method == "ordinal":

        oe = OrdinalEncoder()

        df[cat_cols] = oe.fit_transform(df[cat_cols])

    elif method == "target":

        if target_column is None:

            raise ValueError("target_column required for target encoding")

        for col in cat_cols:

            means = df.groupby(col)[target_column].mean()

            df[col] = df[col].map(means)

    else:

        print("Invalid encoding method")

    print("\nEncoding done using:", method)

    return df