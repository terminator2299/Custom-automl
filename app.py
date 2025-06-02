import streamlit as st
import pandas as pd
import joblib
import os
from automl import preprocessing

st.title("AutoML Prediction App")

# Load artifacts once
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = "outputs/best_model.pkl"
    preprocessor_path = "outputs/preprocessor.pkl"
    feature_cols_path = "outputs/feature_columns.pkl"

    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(feature_cols_path)):
        st.error("Trained model, preprocessor or feature columns not found. Please run training first.")
        return None, None, None

    model = joblib.load(model_path)
    preprocessor_loaded = joblib.load(preprocessor_path)
    feature_columns = joblib.load(feature_cols_path)
    return model, preprocessor_loaded, feature_columns

model, preprocessor_loaded, feature_columns = load_artifacts()

if model is None:
    st.stop()

# Assign loaded preprocessor globally for preprocessing module
preprocessing.preprocessor = preprocessor_loaded

uploaded_file = st.file_uploader("Upload your CSV file for prediction", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Sample", df.head())

    # Let user select target column if it exists in data to drop for prediction
    target_column = None
    cols = df.columns.tolist()
    if len(cols) > 0:
        target_column = st.selectbox("Select target column (if present) to exclude from prediction", options=[None] + cols)

    try:
        # Drop target column if selected
        if target_column and target_column in df.columns:
            df = df.drop(columns=[target_column])

        # Check for missing columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            st.error(f"Input data is missing these required columns: {missing_cols}")
        else:
            # Reorder columns to match training
            df = df[feature_columns]

            # Preprocess
            X_processed = preprocessing.preprocess_data(df, training=False)

            # Predict
            preds = model.predict(X_processed)

            # Show results
            df["Prediction"] = preds
            st.write("### Predictions")
            st.dataframe(df)

            # Download link for predictions CSV
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='predictions.csv',
                mime='text/csv',
            )
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
else:
    st.info("Please upload a CSV file to get started.")
