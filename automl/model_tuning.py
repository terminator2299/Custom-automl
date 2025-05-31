# automl/model_tuning.py

import optuna
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import cross_val_score

def tune_xgboost(X, y):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **params)
        return cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("✅ Best XGBoost params:", study.best_params)
    return XGBClassifier(use_label_encoder=False, eval_metric='logloss', **study.best_params)

def tune_lightgbm(X, y):
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        }
        model = LGBMClassifier(**params)
        return cross_val_score(model, X, y, cv=3, scoring="accuracy").mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)
    print("✅ Best LightGBM params:", study.best_params)
    return LGBMClassifier(**study.best_params)

def tune_hyperparameters(X, y, models_dict):
    print("🎯 Starting hyperparameter tuning...")
    tuned_models = {}

    for name, model in models_dict.items():
        if name == "XGBoost":
            tuned_models[name] = tune_xgboost(X, y)
        elif name == "LightGBM":
            tuned_models[name] = tune_lightgbm(X, y)
        else:
            tuned_models[name] = model  # Keep others unchanged

    # Fit tuned models
    for name, model in tuned_models.items():
        print(f"🔧 Fitting {name} with tuned hyperparameters...")
        model.fit(X, y)

    print("✅ Hyperparameter tuning complete.")
    return tuned_models
