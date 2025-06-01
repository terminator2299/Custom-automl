import pandas as pd
import argparse
import joblib
import os
from automl import preprocessing

def main(input_file):
    print("📥 Loading new data...")
    new_data = pd.read_csv(input_file)

    # Load best model and preprocessor
    model_path = "outputs/best_model.pkl"
    preprocessor_path = "outputs/preprocessor.pkl"

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        print("❌ Trained model or preprocessor not found. Run main.py first.")
        return

    print("📦 Loading best model and preprocessor...")
    model = joblib.load(model_path)
    preprocessor_loaded = joblib.load(preprocessor_path)
    preprocessing.preprocessor = preprocessor_loaded

    # Make sure target column is NOT present in new_data for prediction
    # Just preprocess features
    X_test = preprocessing.preprocess_data(new_data, training=False)

    print("🤖 Making predictions...")
    predictions = model.predict(X_test)

    output_df = new_data.copy()
    output_df["Prediction"] = predictions

    os.makedirs("outputs", exist_ok=True)
    output_df.to_csv("outputs/predictions.csv", index=False)
    print("✅ Predictions saved to outputs/predictions.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    args = parser.parse_args()

    main(args.input)
