# automl/model_training.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def train_models(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(n_estimators=100),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        print(f"🔧 Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model

    print("✅ All base models trained.")
    return trained_models
