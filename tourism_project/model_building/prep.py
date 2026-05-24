# for data manipulation
import pandas as pd
import numpy as np
# for creating folders / reading env vars
import os
# for splitting the data
from sklearn.model_selection import train_test_split
# for hugging face authentication and uploads
from huggingface_hub import HfApi

# Authenticate using the HF_TOKEN environment variable
api = HfApi(token=os.getenv("HF_TOKEN"))

# TODO: replace <your-hf-username> below
# Load the dataset directly from the Hugging Face dataset space
DATASET_PATH = "hf://datasets/Kman42/wellness-tourism-dataset/tourism.csv"
df = pd.read_csv(DATASET_PATH)
print("Dataset loaded successfully. Shape:", df.shape)

# ---------- Data Cleaning ----------
# Drop the unique identifier (not useful for modeling)
if "CustomerID" in df.columns:
    df.drop(columns=["CustomerID"], inplace=True)

# Some datasets contain an unnamed index column from a previous export; drop it if present
df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

# Fix a known data-entry typo in Gender ("Fe Male" -> "Female")
if "Gender" in df.columns:
    df["Gender"] = df["Gender"].replace({"Fe Male": "Female"})

# Define the target column
target_col = "ProdTaken"

# Identify categorical vs numeric feature columns (excluding the target)
categorical_features = [
    "TypeofContact", "Occupation", "Gender",
    "MaritalStatus", "Designation", "ProductPitched"
]
# Keep only those that actually exist in the dataframe
categorical_features = [c for c in categorical_features if c in df.columns]
numeric_features = [
    c for c in df.columns
    if c not in categorical_features + [target_col]
]

# Impute missing values: median for numeric, mode for categorical
for col in numeric_features:
    df[col] = df[col].fillna(df[col].median())
for col in categorical_features:
    df[col] = df[col].fillna(df[col].mode()[0])

print("Missing values after imputation:", int(df.isna().sum().sum()))

# ---------- Train/Test Split ----------
X = df.drop(columns=[target_col])
y = df[target_col]

# Stratify on the target to preserve class balance (target is imbalanced)
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Save locally
Xtrain.to_csv("Xtrain.csv", index=False)
Xtest.to_csv("Xtest.csv", index=False)
ytrain.to_csv("ytrain.csv", index=False)
ytest.to_csv("ytest.csv", index=False)
print("Train/test files saved locally.")

# ---------- Upload train/test files back to the dataset space ----------
files = ["Xtrain.csv", "Xtest.csv", "ytrain.csv", "ytest.csv"]
for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=file_path.split("/")[-1],
        repo_id="Kman42/wellness-tourism-dataset",
        repo_type="dataset",
    )
print("Train/test files uploaded to the Hugging Face Hub.")
