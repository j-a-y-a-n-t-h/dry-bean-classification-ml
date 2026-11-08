"""
Dry Bean multiclass classification project
Run from this folder:
    python train_models.py --data Dry_Bean_Dataset.xlsx

This script:
1. Loads the official Dry Bean dataset.
2. Makes one stratified train/test split.
3. Fits five classifiers using training data only.
4. Evaluates all models on the same held-out test set.
5. Saves models, preprocessing pipelines, test_data.csv and metrics.csv.
"""
import argparse
import os
import json
import time
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

RANDOM_STATE = 42
TARGET = "Class"
MODEL_DIR = "model"
TEST_FILE = "test_data.csv"


def load_dataset(path):
    """Load CSV or Excel input and validate the expected target/features."""
    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    elif path.lower().endswith(".csv"):
        df = pd.read_csv(path)
    else:
        raise ValueError("Dataset must be .xlsx, .xls or .csv")

    df.columns = [str(c).strip() for c in df.columns]
    if TARGET not in df.columns:
        raise ValueError(f"Target column '{TARGET}' was not found.")
    if df.isna().any().any():
        raise ValueError("Dataset contains missing values. Inspect and handle them before training.")
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(str)
    if X.shape[1] < 12 or len(df) < 500:
        raise ValueError("Dataset does not meet assignment minimums (12 features, 500 rows).")
    if not all(pd.api.types.is_numeric_dtype(X[c]) for c in X.columns):
        raise ValueError("All input features are expected to be numeric.")
    return df, X, y


def make_models():
    """Pipelines keep scaling fitted on training data only."""
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", DecisionTreeClassifier(
                max_depth=None, min_samples_split=2, random_state=RANDOM_STATE
            ))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5, weights="distance"))
        ]),
        "Gaussian Naive Bayes": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", GaussianNB())
        ]),
        "Support Vector Machine": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(kernel="rbf", C=3.0, gamma="scale", probability=True,
                               random_state=RANDOM_STATE))
        ]),
    }


def calculate_metrics(model, X_test, y_test, class_labels):
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    # Explicit label order ensures consistent multiclass AUC calculation.
    auc = roc_auc_score(
        y_test, probabilities, labels=class_labels,
        multi_class="ovr", average="macro"
    )
    return {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": auc,
        "Precision_macro": precision_score(y_test, predictions, average="macro", zero_division=0),
        "Recall_macro": recall_score(y_test, predictions, average="macro", zero_division=0),
        "F1_macro": f1_score(y_test, predictions, average="macro", zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Dry_Bean_Dataset.xlsx",
                        help="Path to the UCI Dry Bean Excel or CSV file")
    args = parser.parse_args()

    os.makedirs(MODEL_DIR, exist_ok=True)
    df, X, y = load_dataset(args.data)
    class_labels = sorted(y.unique().tolist())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Dataset: {df.shape[0]} rows, {X.shape[1]} input features")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"Classes ({len(class_labels)}): {class_labels}")

    # Store the held-out test features AND true target for the Streamlit evaluation app.
    test_frame = X_test.copy()
    test_frame[TARGET] = y_test.values
    test_frame.to_csv(TEST_FILE, index=False)

    results = []
    models = make_models()
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        start = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - start

        metrics = calculate_metrics(model, X_test, y_test, class_labels)
        results.append({"Model": name, **metrics, "Training_time_seconds": elapsed})

        safe_name = {
            "Logistic Regression": "logistic_regression",
            "Decision Tree": "decision_tree",
            "KNN": "knn",
            "Gaussian Naive Bayes": "naive_bayes",
            "Support Vector Machine": "svm",
        }[name]
        joblib.dump(model, os.path.join(MODEL_DIR, f"{safe_name}.joblib"))
        print("  " + " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items()))
        print(f"  Training time: {elapsed:.2f} seconds")

    results_df = pd.DataFrame(results)
    results_df.to_csv("metrics.csv", index=False)
    with open(os.path.join(MODEL_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump({
            "feature_columns": X.columns.tolist(),
            "class_labels": class_labels,
            "target_column": TARGET,
            "random_state": RANDOM_STATE,
            "test_size": 0.20
        }, f, indent=2)

    print("\nFINAL MODEL COMPARISON")
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved five models in ./model/")
    print(f"Saved held-out test data to ./{TEST_FILE}")
    print("Saved metric table to ./metrics.csv")


if __name__ == "__main__":
    main()
