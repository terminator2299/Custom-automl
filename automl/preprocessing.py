import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

preprocessor = None  # Global preprocessor

def preprocess_data(df: pd.DataFrame, target_column: str = None, training: bool = True):
    global preprocessor

    # Only require target_column if training=True
    if training and target_column is None:
        raise ValueError("Please specify the target column name during training.")

    if training:
        # Drop rows with missing target
        df = df.dropna(subset=[target_column])

        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Identify feature types
        numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

        # Pipelines for numeric and categorical features
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler())
        ])

        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ])

        preprocessor = ColumnTransformer([
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ])

        X_processed = preprocessor.fit_transform(X)

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_processed, y, test_size=0.2, random_state=42
        )
        
        # Get feature columns
        feature_columns = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test, feature_columns

    else:
        # Prediction mode: target_column may be None or ignored
        X = df.copy()

        if preprocessor is None:
            raise RuntimeError("Preprocessor not initialized. Run training first and load the preprocessor.")

        X_processed = preprocessor.transform(X)
        return X_processed
