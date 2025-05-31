# automl/evaluator.py

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

def evaluate_models(models: dict, X_test, y_test):
    results = []

    print("📈 Evaluating models...")
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results.append({
            "Model": name,
            "Accuracy": acc,
            "F1 Score": f1,
            "ROC AUC": roc_auc if roc_auc is not None else "N/A"
        })

    # Create leaderboard
    leaderboard = pd.DataFrame(results).sort_values(by="F1 Score", ascending=False)
    os.makedirs("outputs", exist_ok=True)
    leaderboard.to_csv("outputs/leaderboard.csv", index=False)

    print("🏆 Leaderboard saved to outputs/leaderboard.csv")
    print(leaderboard)

    # Return best model (based on F1 Score)
    best_model_name = leaderboard.iloc[0]["Model"]
    print(f"🥇 Best model: {best_model_name}")
    return models[best_model_name]
