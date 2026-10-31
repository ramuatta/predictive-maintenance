import os
import json
import argparse
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier

DATA_PATH = "data/engine_data.csv"
OUTPUT_DIR = "outputs"
MODEL_DIR = "models"
TARGET_COLUMN = "Engine Condition"
FEATURE_COLUMNS = ["Engine rpm", "Lub oil pressure", "Fuel pressure", "Coolant pressure", "lub oil temp", "Coolant temp"]


def validate_data(data):
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if data[required].isnull().any().any():
        raise ValueError("Missing values found in required columns.")
    print("Dataset shape:", data.shape)
    print("Missing values in required columns: 0")
    print("Target distribution:")
    print(data[TARGET_COLUMN].value_counts())


def prepare_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = pd.read_csv(DATA_PATH)
    validate_data(data)
    X, y = data[FEATURE_COLUMNS], data[TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    train_data, test_data = X_train.copy(), X_test.copy()
    train_data[TARGET_COLUMN] = y_train
    test_data[TARGET_COLUMN] = y_test
    train_data.to_csv(f"{OUTPUT_DIR}/train_data.csv", index=False)
    test_data.to_csv(f"{OUTPUT_DIR}/test_data.csv", index=False)
    print(f"Training data saved: {OUTPUT_DIR}/train_data.csv ({len(train_data)} rows)")
    print(f"Test data saved: {OUTPUT_DIR}/test_data.csv ({len(test_data)} rows)")


def train_models():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)
    train_path, test_path = f"{OUTPUT_DIR}/train_data.csv", f"{OUTPUT_DIR}/test_data.csv"
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Train/test CSVs missing. Run --prepare-only and download the artifact first.")
    train_data, test_data = pd.read_csv(train_path), pd.read_csv(test_path)
    validate_data(train_data)
    validate_data(test_data)
    X_train, y_train = train_data[FEATURE_COLUMNS], train_data[TARGET_COLUMN]
    X_test, y_test = test_data[FEATURE_COLUMNS], test_data[TARGET_COLUMN]
    print("Loaded training rows:", len(train_data))
    print("Loaded test rows:", len(test_data))

    # Keep the original candidate models and parameter grids unchanged.
    models_and_grids = {
        "Decision Tree": (DecisionTreeClassifier(random_state=42), {
            "max_depth": [3, 5, 10, None], "min_samples_split": [2, 10],
            "min_samples_leaf": [1, 5], "criterion": ["gini", "entropy"]}),
        "Random Forest": (RandomForestClassifier(random_state=42), {
            "n_estimators": [100, 200], "max_depth": [10, None], "min_samples_leaf": [1, 5]}),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), {
            "n_estimators": [100, 200], "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5], "min_samples_leaf": [1, 5]}),
        "AdaBoost": (AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42), {
            "n_estimators": [100, 200], "learning_rate": [0.5, 1.0],
            "estimator__min_samples_leaf": [1, 5]}),
        "XGBoost": (XGBClassifier(random_state=42, eval_metric="logloss", n_jobs=1), {
            "n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 5]}),
    }

    experiment_rows, best_overall_model = [], None
    best_overall_score, best_overall_name = -1, None
    for name, (estimator, param_grid) in models_and_grids.items():
        print(f"\nTuning {name}...")
        search = GridSearchCV(estimator=estimator, param_grid=param_grid,
                              scoring="f1_macro", cv=5, n_jobs=-1, verbose=1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        predictions = model.predict(X_test)
        metrics = {
            "Test Accuracy": accuracy_score(y_test, predictions),
            "Test Precision (macro)": precision_score(y_test, predictions, average="macro", zero_division=0),
            "Test Recall (macro)": recall_score(y_test, predictions, average="macro", zero_division=0),
            "Test F1-score (macro)": f1_score(y_test, predictions, average="macro", zero_division=0),
        }
        experiment_rows.append({
            "Model": f"Tuned {name}", "Selection Metric": "5-fold CV Macro F1",
            "Best CV Macro F1": search.best_score_,
            "Best Parameters": json.dumps(search.best_params_), **metrics,
        })
        model_filename = name.lower().replace(" ", "_")
        joblib.dump(model, f"{MODEL_DIR}/tuned_{model_filename}.joblib")
        pd.DataFrame(search.cv_results_).to_csv(
            f"{OUTPUT_DIR}/{model_filename}_tuning_results.csv", index=False)
        print("Best parameters:", search.best_params_)
        print("Best CV macro F1:", round(search.best_score_, 4))
        print("Test macro F1:", round(metrics["Test F1-score (macro)"], 4))
        if search.best_score_ > best_overall_score:
            best_overall_score, best_overall_model, best_overall_name = (
                search.best_score_, model, name)

    summary = pd.DataFrame(experiment_rows).sort_values("Best CV Macro F1", ascending=False)
    summary.to_csv(f"{OUTPUT_DIR}/experiment_summary.csv", index=False)
    joblib.dump(best_overall_model, f"{MODEL_DIR}/best_model.joblib")
    with open(f"{MODEL_DIR}/best_model_info.json", "w") as file:
        json.dump({"selected_model": best_overall_name,
                   "selection_metric": "5-fold CV Macro F1",
                   "best_cv_macro_f1": best_overall_score}, file, indent=2)
    print("\nSelected model:", best_overall_name)
    print("Best CV macro F1:", round(best_overall_score, 4))
    print("Saved model: models/best_model.joblib")
    print("Saved experiment summary: outputs/experiment_summary.csv")


def main():
    parser = argparse.ArgumentParser(description="Predictive maintenance pipeline")
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-only", action="store_true", help="Validate data and create train/test CSVs")
    modes.add_argument("--train-only", action="store_true", help="Train models using prepared train/test CSVs")
    args = parser.parse_args()
    if args.prepare_only:
        prepare_data()
    else:
        train_models()


if __name__ == "__main__":
    main()
