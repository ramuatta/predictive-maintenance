
import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier


# Project paths
DATA_PATH = "data/engine_data.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
TARGET_COLUMN = "Engine Condition"

FEATURE_COLUMNS = [
    "Engine rpm",
    "Lub oil pressure",
    "Fuel pressure",
    "Coolant pressure",
    "lub oil temp",
    "Coolant temp",
]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # 1. Load and validate the dataset
    data = pd.read_csv(DATA_PATH)

    required_columns = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing_columns = [c for c in required_columns if c not in data.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if data[required_columns].isnull().any().any():
        raise ValueError("Missing values found in required columns.")

    print("Dataset shape:", data.shape)
    print("Missing values in required columns: 0")
    print("Target distribution:")
    print(data[TARGET_COLUMN].value_counts())

    # 2. Split into training and testing data
    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    train_data = X_train.copy()
    train_data[TARGET_COLUMN] = y_train
    test_data = X_test.copy()
    test_data[TARGET_COLUMN] = y_test

    train_data.to_csv(f"{OUTPUT_DIR}/train_data.csv", index=False)
    test_data.to_csv(f"{OUTPUT_DIR}/test_data.csv", index=False)

    # 3. Define model search spaces
    models_and_grids = {
        "Decision Tree": (
            DecisionTreeClassifier(random_state=42),
            {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 10],
                "min_samples_leaf": [1, 5],
                "criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "max_depth": [10, None],
                "min_samples_leaf": [1, 5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
                "min_samples_leaf": [1, 5],
            },
        ),
        "AdaBoost": (
            AdaBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=1, random_state=42),
                random_state=42,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.5, 1.0],
                "estimator__min_samples_leaf": [1, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                n_jobs=1,
            ),
            {
                "n_estimators": [100, 200],
                "learning_rate": [0.05, 0.1],
                "max_depth": [3, 5],
            },
        ),
    }

    experiment_rows = []
    best_overall_model = None
    best_overall_score = -1
    best_overall_name = None

    # 4. Tune and evaluate each model
    for name, (estimator, param_grid) in models_and_grids.items():
        print(f"\nTuning {name}...")

        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring="f1_macro",
            cv=5,
            n_jobs=-1,
            verbose=1,
        )
        search.fit(X_train, y_train)

        model = search.best_estimator_
        predictions = model.predict(X_test)

        metrics = {
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Test Precision (macro)": precision_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Test Recall (macro)": recall_score(
                y_test, predictions, average="macro", zero_division=0
            ),
            "Test F1-score (macro)": f1_score(
                y_test, predictions, average="macro", zero_division=0
            ),
        }

        row = {
            "Model": f"Tuned {name}",
            "Selection Metric": "5-fold CV Macro F1",
            "Best CV Macro F1": search.best_score_,
            "Best Parameters": json.dumps(search.best_params_),
            **metrics,
        }
        experiment_rows.append(row)

        model_filename = name.lower().replace(" ", "_")
        joblib.dump(model, f"{MODEL_DIR}/tuned_{model_filename}.joblib")

        pd.DataFrame(search.cv_results_).to_csv(
            f"{OUTPUT_DIR}/{model_filename}_tuning_results.csv",
            index=False,
        )

        print("Best parameters:", search.best_params_)
        print("Best CV macro F1:", round(search.best_score_, 4))
        print("Test macro F1:", round(metrics["Test F1-score (macro)"], 4))

        if search.best_score_ > best_overall_score:
            best_overall_score = search.best_score_
            best_overall_model = model
            best_overall_name = name

    # 5. Save experiment summary and selected model
    summary = pd.DataFrame(experiment_rows).sort_values(
        "Best CV Macro F1", ascending=False
    )
    summary.to_csv(f"{OUTPUT_DIR}/experiment_summary.csv", index=False)

    joblib.dump(best_overall_model, f"{MODEL_DIR}/best_model.joblib")
    with open(f"{MODEL_DIR}/best_model_info.json", "w") as f:
        json.dump(
            {
                "selected_model": best_overall_name,
                "selection_metric": "5-fold CV Macro F1",
                "best_cv_macro_f1": best_overall_score,
            },
            f,
            indent=2,
        )

    print("\nSelected model:", best_overall_name)
    print("Best CV macro F1:", round(best_overall_score, 4))
    print("Saved model: models/best_model.joblib")
    print("Saved experiment summary: outputs/experiment_summary.csv")


if __name__ == "__main__":
    main()
