import pickle
import os
import random
from classifier import SimpleDecisionTree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Custom Decision Tree Classifier...")
    clf = SimpleDecisionTree(max_depth=4)
    clf.fit(X_train, y_train)
    
    print("Evaluating Model...")
    preds = clf.predict(X_test)
    
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average='weighted', zero_division=0)
    rec = recall_score(y_test, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_test, preds, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, preds, labels=["Junior", "Mid", "Senior"])
    
    print("\n--- Classification Performance ---")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Confusion Matrix (Junior, Mid, Senior):\n{cm}\n")
    
    # model_path = os.path.join(os.path.dirname(__file__), 'decision_tree_classifier.pkl')
    # with open(model_path, 'wb') as f:
    #     pickle.dump(clf, f)
    # print(f"Model saved successfully to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
