import json
import os
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

DATASET_PATH = os.path.join("hotel_synth_dataset", "pairs.jsonl")
MODEL_OUT_PATH = "candidate_ranker.pkl"

def parse_dataset(filepath):
    data = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                record = json.loads(line)
                
                # Extract features
                reasons = record.get("reasons", {})
                req_cov = reasons.get("required_skill_coverage", 0.0)
                pref_cov = reasons.get("preferred_skill_coverage", 0.0)
                
                # Ordinal Encode Experience Fit
                exp_fit_str = reasons.get("experience_fit", "none").lower()
                exp_map = {"none": 0, "low": 1, "meets": 2, "exceeds": 3}
                exp_val = exp_map.get(exp_fit_str, 0)
                
                # Encode Department Fit (boolean)
                dept_fit = 1.0 if reasons.get("department_fit", False) else 0.0
                
                # Target
                target = float(record.get("match_score_true", 0.0))
                
                data.append({
                    "req_coverage": float(req_cov),
                    "pref_coverage": float(pref_cov),
                    "experience_fit": float(exp_val),
                    "department_fit": dept_fit,
                    "target": target
                })
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None
        
    return pd.DataFrame(data)

def train_model():
    print(f"Loading dataset from {DATASET_PATH}...")
    df = parse_dataset(DATASET_PATH)
    
    if df is None or df.empty:
        print("Dataset is empty or could not be loaded. Cannot train.")
        return
        
    print(f"Loaded {len(df)} records.")
    
    X = df[["req_coverage", "pref_coverage", "experience_fit", "department_fit"]]
    y = df["target"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Random Forest Regressor...")
    model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating Model...")
    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    
    print(f"Test Set MAE: {mae:.2f} points (out of 100)")
    print(f"Test Set R²:  {r2:.3f}")
    
    print("\nFeature Importances:")
    importances = model.feature_importances_
    features = X.columns
    for f, imp in zip(features, importances):
        print(f" - {f}: {imp*100:.1f}%")
        
    print(f"\nSaving model to {MODEL_OUT_PATH}...")
    #joblib.dump(model, MODEL_OUT_PATH)
    print("Training complete and model saved!")

if __name__ == "__main__":
    train_model()
