import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features


# load dataset
df = pd.read_csv("data/sample.csv")

print("Dataset Loaded Successfully")

# analyze dataset
analyze_data(df)

# missing values
df = handle_missing_values(df, method="median")

# encoding
df = encode_categorical(
        df,
        method="onehot",
        target_column="target"  # only needed for target encoding
)

# scaling
df = scale_features(df, method="standard")

print("\nFinal Preprocessed Dataset:\n")

print(df)