import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from feature_selector import FeatureSelector

# load dataset
df = pd.read_csv("data/sample.csv")

print("Dataset Loaded Successfully")

# analyze dataset
analyze_data(df)

# handle missing values
df = handle_missing_values(df)

# encode categorical variables
df = encode_categorical(df)

# scale features
df = scale_features(df)

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