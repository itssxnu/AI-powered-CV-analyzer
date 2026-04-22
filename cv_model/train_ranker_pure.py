import json
import os
import random
import pickle
from classifier import SimpleDecisionTreeRegressor

DATASET_PATH = os.path.join(os.path.dirname(__file__), "hotel_synth_dataset", "pairs.jsonl")
MODEL_OUT_PATH = os.path.join(os.path.dirname(__file__), "candidate_ranker_pure.pkl")

def parse_dataset(filepath):
    X = []
    y = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                reasons = record.get("reasons", {})
                req_cov = float(reasons.get("required_skill_coverage", 0.0))
                pref_cov = float(reasons.get("preferred_skill_coverage", 0.0))
                
                exp_fit_str = reasons.get("experience_fit", "none").lower()
                exp_map = {"none": 0, "low": 1, "meets": 2, "exceeds": 3}
                exp_val = float(exp_map.get(exp_fit_str, 0))
                
                dept_fit = 1.0 if reasons.get("department_fit", False) else 0.0
                
                target = float(record.get("match_score_true", 0.0))
                
                X.append([req_cov, pref_cov, exp_val, dept_fit])
                y.append(target)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None
        
    return X, y

def train_and_evaluate():
    print(f"Loading dataset from {DATASET_PATH}...")
    X, y = parse_dataset(DATASET_PATH)
    
    if not X:
        print("Dataset is empty.")
        return
        
    print(f"Loaded {len(X)} records.")
    
    # Train/Test Split (80/20)
    data = list(zip(X, y))
    random.seed(42)
    random.shuffle(data)
    
    split_idx = int(len(data) * 0.8)
    train_data = data[:split_idx]
    test_data = data[split_idx:]
    
    X_train, y_train = zip(*train_data)
    X_test, y_test = zip(*test_data)
    
    print("Training Pure Python Decision Tree Regressor (max_depth=8)...")
    # max_depth 8 gives a reasonable fit for 6000 points without overfitting or taking too long
    model = SimpleDecisionTreeRegressor(max_depth=8, min_samples_split=5)
    model.fit(list(X_train), list(y_train))
    
    print("Evaluating Model...")
    preds = model.predict(list(X_test))
    
    # Calculate MAE
    mae = sum(abs(t - p) for t, p in zip(y_test, preds)) / len(y_test)
    
    # Calculate R2
    mean_y = sum(y_test) / len(y_test)
    ss_tot = sum((t - mean_y)**2 for t in y_test)
    ss_res = sum((t - p)**2 for t, p in zip(y_test, preds))
    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    print(f"\nRegression Metrics:")
    print(f"Test Set MAE: {mae:.2f} points (out of 100)")
    print(f"Test Set R²:  {r2:.3f}")
    
    # Calculate Classification Metrics (Threshold 75)
    print("\n" + "="*50)
    print("CLASSIFICATION METRICS (Threshold >= 75 for 'Good Match')")
    print("="*50)
    
    THRESHOLD = 75.0
    y_true_cls = [1 if t >= THRESHOLD else 0 for t in y_test]
    y_pred_cls = [1 if p >= THRESHOLD else 0 for p in preds]
    
    # Accuracy
    correct = sum(1 for yt, yp in zip(y_true_cls, y_pred_cls) if yt == yp)
    acc = correct / len(y_true_cls)
    
    # Confusion Matrix
    cm = {0: {0:0, 1:0}, 1: {0:0, 1:0}}
    for yt, yp in zip(y_true_cls, y_pred_cls):
        cm[yt][yp] += 1
        
    tp = cm[1][1]
    fp = cm[0][1]
    fn = cm[1][0]
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"Classification Threshold: {THRESHOLD} points")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {precision:.4f} (For Good Match)")
    print(f"Recall:    {recall:.4f} (For Good Match)")
    print(f"F1 Score:  {f1:.4f} (For Good Match)")
    
    print("\nConfusion Matrix (Rows: Actual, Cols: Predicted):")
    print(f"{'':>15} | {'Pred Not Match':>14} | {'Pred Match':>10}")
    print("-" * 45)
    print(f"Act Not Match (<75)| {cm[0][0]:>14} | {cm[0][1]:>10}")
    print(f"Act Match (>=75)   | {cm[1][0]:>14} | {cm[1][1]:>10}")
    
    # print(f"\nSaving model to {MODEL_OUT_PATH}...")
    # with open(MODEL_OUT_PATH, 'wb') as f:
    #     pickle.dump(model, f)
    # print("Done!")

if __name__ == "__main__":
    train_and_evaluate()
