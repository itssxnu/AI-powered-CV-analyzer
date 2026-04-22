import re
from datetime import datetime
from typing import Dict, Any

def _parse_duration_to_years(duration_str: str) -> float:
    """
    Attempts to parse a duration string into a float representing years.
    Handles varied formats from raw CV text or LLM extraction, for example:
    - "Jan 2018 - Present"
    - "2015 - 2019"
    - "2 years"
    - "1 year 6 months"
    """
    if not duration_str:
        return 0.0
        
    duration_str = duration_str.lower().strip()
    
    # Check explicit "X years Y months" format first
    year_match = re.search(r'(\d+)\s*y', duration_str)
    month_match = re.search(r'(\d+)\s*m', duration_str)
    
    if year_match or month_match:
        years = float(year_match.group(1)) if year_match else 0.0
        months = float(month_match.group(1)) if month_match else 0.0
        return years + (months / 12.0)
        
    # Check absolute years format "2015 - 2019" or "2018 - Present"
    years = re.findall(r'\b(20\d{2}|19\d{2})\b', duration_str)
    
    if len(years) == 2:
        start = int(years[0])
        end = int(years[1])
        return max(0.0, float(end - start))
    elif len(years) == 1 and ("present" in duration_str or "current" in duration_str or "now" in duration_str):
        start = int(years[0])
        current_year = datetime.now().year
        return max(0.0, float(current_year - start))
        
    return 0.0

def calculate_total_experience(cv_json: Any) -> float:
    """
    Iterates through the 'experience' array in the structured CV JSON to determine total YoE.
    """
    if not isinstance(cv_json, dict):
        return 0.0
        
    total_years = 0.0
    
    # 1. First check if there's an explicit profile summary with years
    if "profile" in cv_json and isinstance(cv_json["profile"], dict) and "years_experience" in cv_json["profile"]:
        val = cv_json["profile"]["years_experience"]
        if val is not None:
            try:
                return float(val)
            except:
                pass
                
    # 2. Iterate through experience blocks summing duration
    experience_list = cv_json.get("experience", [])
    if isinstance(experience_list, list):
        for exp in experience_list:
            if isinstance(exp, dict) and "duration" in exp:
                total_years += _parse_duration_to_years(exp["duration"])
                
    return round(total_years, 2)

def evaluate_skill_depth(cv_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Looks for leadership indicators, management titles, and advanced system flags
    to weigh the candidate's operational level.
    """
    flags = []
    
    # Check recent role titles
    exp_list = cv_json.get("experience", [])
    if isinstance(exp_list, list):
        for exp in exp_list:
            if isinstance(exp, dict):
                role = str(exp.get("role", "")).lower()
                desc = str(exp.get("description", "")).lower()
                
                if any(x in role for x in ["director", "head of", "vp", "general manager", "gm"]):
                    if "Executive Leadership" not in flags: flags.append("Executive Leadership")
                elif any(x in role for x in ["manager", "supervisor", "lead", "coordinator"]):
                    if "Management/Supervisory" not in flags: flags.append("Management/Supervisory")
                    
                if "budgeting" in desc or "p&l" in desc or "revenue" in desc:
                    if "Financial Responsibility" not in flags: flags.append("Financial Responsibility")

    # Check hospitality skills subset for compliance/systems indicating seniority
    hosp = cv_json.get("hospitality", {})
    if isinstance(hosp, dict):
        ops = hosp.get("operational_flags", {})
        if isinstance(ops, dict) and ops.get("night_audit_experience"):
            if "Night Audit Operations" not in flags: flags.append("Night Audit Operations")
            
        sys = hosp.get("systems_tools", [])
        if isinstance(sys, list) and any("opera" in str(s).lower() for s in sys):
            if "Enterprise PMS Knowledge" not in flags: flags.append("Enterprise PMS Knowledge")

    return {
        "leadership_flags": flags,
        "is_manager": "Management/Supervisory" in flags or "Executive Leadership" in flags,
        "is_executive": "Executive Leadership" in flags
    }

class SimpleDecisionTree:
    def __init__(self, max_depth=4):
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_features = len(X[0])
        unique_classes = list(set(y))

        if len(unique_classes) == 1 or depth == self.max_depth or num_samples < 2:
            return {'leaf': True, 'value': self._most_common_label(y)}

        best_feat, best_thresh = self._best_split(X, y, num_features)
        
        if best_feat is None:
            return {'leaf': True, 'value': self._most_common_label(y)}

        left_indices, right_indices = self._split(X, best_feat, best_thresh)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return {'leaf': True, 'value': self._most_common_label(y)}

        left_X, left_y = [X[i] for i in left_indices], [y[i] for i in left_indices]
        right_X, right_y = [X[i] for i in right_indices], [y[i] for i in right_indices]

        left_branch = self._build_tree(left_X, left_y, depth + 1)
        right_branch = self._build_tree(right_X, right_y, depth + 1)

        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': left_branch,
            'right': right_branch
        }

    def _best_split(self, X, y, num_features):
        best_gini = 1.0
        best_feat, best_thresh = None, None

        for feat_idx in range(num_features):
            thresholds = list(set([row[feat_idx] for row in X]))
            for thresh in thresholds:
                left_indices, right_indices = self._split(X, feat_idx, thresh)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                gini = self._gini_impurity([y[i] for i in left_indices], [y[i] for i in right_indices])
                
                if gini < best_gini:
                    best_gini = gini
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _split(self, X, feat_idx, thresh):
        left_indices = [i for i, row in enumerate(X) if row[feat_idx] <= thresh]
        right_indices = [i for i, row in enumerate(X) if row[feat_idx] > thresh]
        return left_indices, right_indices

    def _gini_impurity(self, left_y, right_y):
        def _gini(y_subset):
            size = len(y_subset)
            if size == 0: return 0
            counts = {}
            for label in y_subset:
                counts[label] = counts.get(label, 0) + 1
            return 1.0 - sum((count / size) ** 2 for count in counts.values())

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * _gini(left_y) + (len(right_y) / n) * _gini(right_y)

    def _most_common_label(self, y):
        counts = {}
        for label in y:
            counts[label] = counts.get(label, 0) + 1
        return max(counts, key=counts.get)

    def predict(self, X):
        return [self._predict(inputs, self.tree) for inputs in X]

    def _predict(self, inputs, node):
        if node['leaf']:
            return node['value']
        if inputs[node['feature']] <= node['threshold']:
            return self._predict(inputs, node['left'])
        return self._predict(inputs, node['right'])


class SimpleDecisionTreeRegressor:
    def __init__(self, max_depth=4, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def fit(self, X, y):
        self.tree = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        num_samples = len(y)
        num_features = len(X[0])
        
        # Calculate variance to see if we should stop
        mean_y = sum(y) / num_samples if num_samples > 0 else 0
        variance = sum((val - mean_y) ** 2 for val in y) / num_samples if num_samples > 0 else 0

        if variance == 0 or depth == self.max_depth or num_samples < self.min_samples_split:
            return {'leaf': True, 'value': mean_y}

        best_feat, best_thresh = self._best_split(X, y, num_features)
        
        if best_feat is None:
            return {'leaf': True, 'value': mean_y}

        left_indices, right_indices = self._split(X, best_feat, best_thresh)
        
        if len(left_indices) == 0 or len(right_indices) == 0:
            return {'leaf': True, 'value': mean_y}

        left_X, left_y = [X[i] for i in left_indices], [y[i] for i in left_indices]
        right_X, right_y = [X[i] for i in right_indices], [y[i] for i in right_indices]

        left_branch = self._build_tree(left_X, left_y, depth + 1)
        right_branch = self._build_tree(right_X, right_y, depth + 1)

        return {
            'leaf': False,
            'feature': best_feat,
            'threshold': best_thresh,
            'left': left_branch,
            'right': right_branch
        }

    def _best_split(self, X, y, num_features):
        best_mse = float('inf')
        best_feat, best_thresh = None, None

        for feat_idx in range(num_features):
            # To optimize, we don't need to test every single value if dataset is huge, 
            # but for our dataset size testing unique values is fine.
            thresholds = list(set([row[feat_idx] for row in X]))
            for thresh in thresholds:
                left_indices, right_indices = self._split(X, feat_idx, thresh)
                if len(left_indices) == 0 or len(right_indices) == 0:
                    continue
                
                left_y = [y[i] for i in left_indices]
                right_y = [y[i] for i in right_indices]
                
                mse = self._calculate_mse_split(left_y, right_y)
                
                if mse < best_mse:
                    best_mse = mse
                    best_feat = feat_idx
                    best_thresh = thresh

        return best_feat, best_thresh

    def _split(self, X, feat_idx, thresh):
        left_indices = [i for i, row in enumerate(X) if row[feat_idx] <= thresh]
        right_indices = [i for i, row in enumerate(X) if row[feat_idx] > thresh]
        return left_indices, right_indices

    def _calculate_mse_split(self, left_y, right_y):
        def _mse(y_subset):
            size = len(y_subset)
            if size == 0: return 0
            mean = sum(y_subset) / size
            return sum((val - mean) ** 2 for val in y_subset) / size

        n = len(left_y) + len(right_y)
        return (len(left_y) / n) * _mse(left_y) + (len(right_y) / n) * _mse(right_y)

    def predict(self, X):
        return [self._predict(inputs, self.tree) for inputs in X]

    def _predict(self, inputs, node):
        if node['leaf']:
            return node['value']
        if inputs[node['feature']] <= node['threshold']:
            return self._predict(inputs, node['left'])
        return self._predict(inputs, node['right'])


import os
import pickle

_DT_MODEL = None

def get_dt_model():
    global _DT_MODEL
    if _DT_MODEL is None:
        model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_classifier.pkl')
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                _DT_MODEL = pickle.load(f)
        else:
            # Fallback if model not trained yet
            _DT_MODEL = SimpleDecisionTree()
    return _DT_MODEL

def classify_candidate(cv_json: Dict[str, Any]) -> Dict[str, Any]:
    """
    Takes a parsed CV JSON dictionary.
    Calculates YoE and Skill Depth, then uses the ML Decision Tree to classify level.
    Levels: "Junior", "Mid", "Senior"
    """
    if not cv_json or not isinstance(cv_json, dict):
        return {"level": "Unknown", "yoe": 0.0, "explanation": "Invalid CV JSON payload."}

    yoe = calculate_total_experience(cv_json)
    depth = evaluate_skill_depth(cv_json)
    
    is_manager = 1 if depth["is_manager"] else 0
    is_executive = 1 if depth["is_executive"] else 0
    
    features = [[yoe, is_manager, is_executive]]
    model = get_dt_model()
    
    # Predict level using tree
    if model.tree is None:
        # Fallback if no tree trained
        if yoe >= 7.0 or depth["is_executive"]:
            level = "Senior"
        elif yoe >= 3.0 or depth["is_manager"]:
            level = "Mid"
        else:
            level = "Junior"
    else:
        level = model.predict(features)[0]
    
    # Generate ML-style explanation based on features
    explanation = f"Decision Tree assigned {level} level based on YoE: {yoe}, Manager: {bool(is_manager)}, Executive: {bool(is_executive)}."

    return {
        "level": level,
        "yoe": yoe,
        "leadership_flags": depth["leadership_flags"],
        "explanation": explanation.strip()
    }
