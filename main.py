import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector

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


# Separate features and target
target_column = "target"  # change this
X = df.drop(columns=[target_column])
y = df[target_column]

# Feature Selection
selector = FeatureSelector(task_type="classification")
X_selected = selector.auto_select(X, y)

print("Final selected features:")
print(X_selected.head())