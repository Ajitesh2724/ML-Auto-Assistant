import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def encode_categorical(df, method="onehot", target_column=None, max_unique=20):

    df = df.copy()

    # detect categorical columns
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # remove target column
    if target_column in cat_cols:
        cat_cols.remove(target_column)

    # split cardinality
    low_card_cols = [col for col in cat_cols if df[col].nunique() <= max_unique]
    high_card_cols = [col for col in cat_cols if df[col].nunique() > max_unique]

    if len(high_card_cols) > 0:
        print(f"⚠️ Skipping high-cardinality columns: {high_card_cols}")

    if not low_card_cols:
        print("No categorical columns to encode")
        return df

    # -------------------------------
    # ONE-HOT
    # -------------------------------
    if method == "onehot":
        df = pd.get_dummies(df, columns=low_card_cols, dtype=int)

    # -------------------------------
    # LABEL
    # -------------------------------
    elif method == "label":
        le = LabelEncoder()
        for col in low_card_cols:
            df[col] = le.fit_transform(df[col])

    # -------------------------------
    # ORDINAL
    # -------------------------------
    elif method == "ordinal":
        oe = OrdinalEncoder()
        df[low_card_cols] = oe.fit_transform(df[low_card_cols])

    # -------------------------------
    # TARGET
    # -------------------------------
    elif method == "target":

        if target_column is None:
            raise ValueError("target_column required for target encoding")

        for col in low_card_cols:
            means = df.groupby(col)[target_column].mean()
            df[col] = df[col].map(means)

    else:
        raise ValueError("Invalid encoding method")

    print("\nEncoding done using:", method)
    print("Encoded columns:", low_card_cols)

    return df