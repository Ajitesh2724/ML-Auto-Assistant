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

# handle missing values
df = handle_missing_values(df)

# encode categorical variables
df = encode_categorical(df)

# scale features
df = scale_features(df)

print("\nFinal Preprocessed Dataset:\n")

print(df)