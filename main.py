
import pandas as pd
from automl.preprocessing import preprocess_data
from automl.model_training import train_models
from automl.model_tuning import tune_hyperparameters
from automl.evaluator import evaluate_models
import joblib
import os

# Set config
DATA_PATH = "data/train.csv"  # Update with your dataset path
TARGET_COLUMN = "Survived"     # Update with your target column

def main():
    print("🔄 Loading dataset...")
    df = pd.read_csv(DATA_PATH)

    print("✅ Preprocessing data...")
    X_train, X_test, y_train, y_test = preprocess_data(df, TARGET_COLUMN)

    print("🤖 Training base models...")
    models = train_models(X_train, y_train)

    print("🔍 Tuning hyperparameters for top models...")
    tuned_models = tune_hyperparameters(X_train, y_train, models)

    print("📊 Evaluating models and saving results...")
    best_model = evaluate_models(tuned_models, X_test, y_test)

    print("💾 Saving best model to disk...")
    os.makedirs("outputs", exist_ok=True)
    joblib.dump(best_model, "outputs/best_model.pkl")
    print("✅ Best model saved as outputs/best_model.pkl")

if __name__ == "__main__":
    main()
