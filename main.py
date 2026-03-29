import pandas as pd

from modules.data_analyzer import analyze_data
from modules.missing_handler import handle_missing_values
from modules.encoder import encode_categorical
from modules.scaler import scale_features
from modules.feature_selector import FeatureSelector
from modules.model_evaluator import evaluate_model
from modules.model_trainer import ModelTrainer


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

# Split dataset into training and testing sets

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(

        X_selected,

        y,

        test_size=0.2,

        random_state=42

)


# Train model using ModelTrainer

trainer = ModelTrainer(

        task_type="classification",     # classification or regression

        model_name="random_forest"      # logistic, random_forest, svm

)

model = trainer.train(

        X_train,

        y_train

)


# Make predictions

y_pred = trainer.predict(

        X_test

)


# Evaluate model performance

results = evaluate_model(

        y_test,

        y_pred,

        problem_type="classification"

)


print("\nEvaluation Results Summary:")

print(results)