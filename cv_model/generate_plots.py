import os
import json
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

def plot_visualizations():
    model_path = os.path.join(os.path.dirname(__file__), 'candidate_ranker.pkl')
    dataset_path = os.path.join(os.path.dirname(__file__), "hotel_synth_dataset", "pairs.jsonl")
    
    model = joblib.load(model_path)
    
    X_list = []
    y_true_scores = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            reasons = record.get("reasons", {})
            
            exp_fit_str = reasons.get("experience_fit", "none").lower()
            exp_map = {"none": 0, "low": 1, "meets": 2, "exceeds": 3}
            
            X_list.append({
                "req_coverage": float(reasons.get("required_skill_coverage", 0.0)),
                "pref_coverage": float(reasons.get("preferred_skill_coverage", 0.0)),
                "experience_fit": float(exp_map.get(exp_fit_str, 0)),
                "department_fit": 1.0 if reasons.get("department_fit", False) else 0.0
            })
            y_true_scores.append(float(record.get("match_score_true", 0.0)))
            
    X_test = pd.DataFrame(X_list)
    y_pred_scores = model.predict(X_test)
    
    THRESHOLD = 75.0
    y_true = [1 if s >= THRESHOLD else 0 for s in y_true_scores]
    y_pred = [1 if s >= THRESHOLD else 0 for s in y_pred_scores]
    
    # Plot Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Not Match', 'Good Match'],
                yticklabels=['Not Match', 'Good Match'])
    plt.title('Confusion Matrix - Candidate Ranker')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig('confusion_matrix.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Plot ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_scores)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) - Candidate Ranker')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png', bbox_inches='tight', dpi=300)
    plt.close()
    print("Plots generated successfully!")

if __name__ == "__main__":
    plot_visualizations()
