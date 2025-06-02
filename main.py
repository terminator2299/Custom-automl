import pandas as pd
from automl import preprocessing
from automl.model_training import train_models
from automl.model_tuning import tune_hyperparameters
from automl.evaluator import evaluate_models
import joblib
import os

DATA_PATH = "data/train.csv"
TARGET_COLUMN = "Survived"

def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("✅ Preprocessing data...")
    X_train, X_test, y_train, y_test, feature_columns = preprocessing.preprocess_data(df, TARGET_COLUMN, training=True)

    print("🤖 Training base models...")
    models = train_models(X_train, y_train)

    print("🔍 Tuning hyperparameters for top models...")
    tuned_models = tune_hyperparameters(X_train, y_train, models)

    print("📊 Evaluating models and selecting best...")
    best_model = evaluate_models(tuned_models, X_test, y_test)

    print("💾 Saving best model, preprocessor, and feature columns to disk...")
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/best_model.pkl")
    joblib.dump(preprocessing.preprocessor, "outputs/preprocessor.pkl")
    joblib.dump(feature_columns, "outputs/feature_columns.pkl")

    print("✅ Artifacts saved: model, preprocessor, and feature columns.")

if __name__ == "__main__":
    main()
