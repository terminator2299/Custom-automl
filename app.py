import streamlit as st
import pandas as pd
import joblib
import os
import sklearn
from automl import preprocessing

st.set_page_config(page_title="AutoML Predictor", layout="wide")
st.title("🚀 AutoML Prediction App")
st.markdown("Upload your CSV file and get instant predictions using a pre-trained machine learning pipeline.")

# Load trained artifacts
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model_path = "outputs/best_model.pkl"
    preprocessor_path = "outputs/preprocessor.pkl"
    feature_cols_path = "outputs/feature_columns.pkl"

    if not (os.path.exists(model_path) and os.path.exists(preprocessor_path) and os.path.exists(feature_cols_path)):
        return None, None, None

    try:
        # Load feature columns first as it's the simplest
        feature_columns = joblib.load(feature_cols_path)
        
        # Load model
        model = joblib.load(model_path)
        
        # Load preprocessor with error handling
        try:
            preprocessor = joblib.load(preprocessor_path)
        except AttributeError as e:
            st.error(f"""
            ❌ Error loading preprocessor: {str(e)}
            
            This error typically occurs when:
            1. The Python version used to save the model is different from the current version
            2. The scikit-learn version is different
            3. The class definitions have changed
            
            Current scikit-learn version: {sklearn.__version__}
            
            Please ensure you're using the same Python and scikit-learn versions as when the model was trained.
            """)
            return None, None, None
            
        return model, preprocessor, feature_columns
    except Exception as e:
        st.error(f"❌ Error loading model artifacts: {str(e)}")
        return None, None, None

model, preprocessor_loaded, feature_columns = load_artifacts()

if not model:
    st.error("❌ Required model files not found. Please run the training script (`main.py`) before using this app.")
    st.stop()

# Assign preprocessor to the module-level variable
preprocessing.preprocessor = preprocessor_loaded

uploaded_file = st.file_uploader("📁 Upload your CSV file for prediction", type=["csv"])

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully.")
        st.subheader("📄 Uploaded Data Preview")
        st.dataframe(df.head())

        # Optional target column exclusion
        possible_cols = df.columns.tolist()
        target_column = st.selectbox(
            "🎯 Select a target column to exclude from prediction (optional)", 
            options=[None] + possible_cols
        )

        if target_column:
            df = df.drop(columns=[target_column])

        # Check for missing columns
        missing_cols = set(feature_columns) - set(df.columns)
        if missing_cols:
            st.warning(f"⚠️ Your data is missing the following required columns: `{', '.join(missing_cols)}`")
            st.stop()

        # Reorder and preprocess
        df = df[feature_columns]
        processed_data = preprocessing.preprocess_data(df, training=False)

        # Prediction
        predictions = model.predict(processed_data)
        df["Prediction"] = predictions

        st.subheader("🔮 Predictions")
        st.dataframe(df)

        # Download predictions
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions as CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"❌ Error during processing or prediction: {e}")
else:
    st.info("👈 Upload a CSV file to begin prediction.")
