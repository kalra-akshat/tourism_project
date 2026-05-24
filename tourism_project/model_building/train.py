# for data manipulation
import pandas as pd
import numpy as np
# preprocessing / pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
# model training, tuning, evaluation
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score)
# model serialization
import joblib
import os
# hugging face authentication / uploads
from huggingface_hub import HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError
import mlflow

# In CI we run an MLflow server locally on port 5000 (see pipeline.yml)
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("mlops-tourism-training-experiment")

api = HfApi(token=os.getenv("HF_TOKEN"))

# TODO: replace <your-hf-username> in the four paths below
Xtrain = pd.read_csv("hf://datasets/Kman42/wellness-tourism-dataset/Xtrain.csv")
Xtest  = pd.read_csv("hf://datasets/Kman42/wellness-tourism-dataset/Xtest.csv")
ytrain = pd.read_csv("hf://datasets/Kman42/wellness-tourism-dataset/ytrain.csv").squeeze()
ytest  = pd.read_csv("hf://datasets/Kman42/wellness-tourism-dataset/ytest.csv").squeeze()

# Identify categorical vs numeric features from the loaded training frame
categorical_features = [c for c in ["TypeofContact","Occupation","Gender",
                                    "MaritalStatus","Designation","ProductPitched"]
                        if c in Xtrain.columns]
numeric_features = [c for c in Xtrain.columns if c not in categorical_features]

# Preprocessor
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features)
)

# Handle class imbalance
neg, pos = (ytrain == 0).sum(), (ytrain == 1).sum()
scale_pos_weight = float(neg) / float(pos)

xgb_model = xgb.XGBClassifier(
    random_state=42, n_jobs=-1, eval_metric="logloss",
    scale_pos_weight=scale_pos_weight
)

# Hyperparameter grid
param_grid = {
    "xgbclassifier__n_estimators": [50, 100, 150],
    "xgbclassifier__max_depth": [3, 5, 7],
    "xgbclassifier__learning_rate": [0.01, 0.05, 0.1],
    "xgbclassifier__subsample": [0.7, 0.8, 1.0],
    "xgbclassifier__colsample_bytree": [0.7, 0.8, 1.0],
}

model_pipeline = make_pipeline(preprocessor, xgb_model)

with mlflow.start_run():
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring="f1")
    grid_search.fit(Xtrain, ytrain)

    # Log each parameter combination as a nested run
    results = grid_search.cv_results_
    for i in range(len(results["params"])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results["params"][i])
            mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])

    # Best model + evaluation
    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    y_pred_train = best_model.predict(Xtrain)
    y_pred_test = best_model.predict(Xtest)
    y_proba_test = best_model.predict_proba(Xtest)[:, 1]

    mlflow.log_metrics({
        "train_accuracy": accuracy_score(ytrain, y_pred_train),
        "test_accuracy": accuracy_score(ytest, y_pred_test),
        "test_precision": precision_score(ytest, y_pred_test),
        "test_recall": recall_score(ytest, y_pred_test),
        "test_f1": f1_score(ytest, y_pred_test),
        "test_roc_auc": roc_auc_score(ytest, y_proba_test),
    })

    # Save the best model locally
    model_path = "best_tourism_model_v1.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Register the model on the Hugging Face model hub
    repo_id = "Kman42/wellness-tourism-model"
    repo_type = "model"
    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Model repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("Best model registered on the Hugging Face model hub.")
