import pickle
import os
import random
from classifier import SimpleDecisionTree

def generate_synthetic_data(num_samples=1000):
    random.seed(42)
    X = []
    y = []
    
    for _ in range(num_samples):
        yoe = random.uniform(0.0, 20.0)
        is_exec = 1 if random.random() < 0.1 else 0
        is_man = 1 if (is_exec == 1 or random.random() < 0.3) else 0
        
        X.append([yoe, is_man, is_exec])
        
        if yoe >= 7.0 or is_exec == 1:
            y.append("Senior")
        elif yoe >= 3.0 or is_man == 1:
            y.append("Mid")
        else:
            y.append("Junior")
            
    return X, y

def train_and_save_model():
    print("Generating synthetic data...")
    X, y = generate_synthetic_data(2000)
    
    print("Training Custom Decision Tree Classifier...")
    clf = SimpleDecisionTree(max_depth=4)
    clf.fit(X, y)
    
    preds = clf.predict(X)
    acc = sum(1 for p, t in zip(preds, y) if p == t) / len(y)
    print(f"Model accuracy on synthetic data: {acc * 100:.2f}%")
    
    model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_classifier.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
