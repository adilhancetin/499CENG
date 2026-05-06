from ucimlrepo import fetch_ucirepo 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support





def evaluate_model(y_true, y_pred, model_name):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    
    print(f"\n{model_name}")
    print(f"accuracy:  {accuracy:.4f}")
    print(f"precision: {precision:.4f}")
    print(f"recall:    {recall:.4f}")
    print(f"f1:        {f1:.4f}")


def main():
    dry_bean = fetch_ucirepo(id=602)
    
    X = dry_bean.data.features
    y = dry_bean.data.targets
    
    if y.shape[1] == 1:
        y = y.squeeze()
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    clf_gini = DecisionTreeClassifier(criterion="gini", random_state=42)
    clf_gini.fit(X_train, y_train)
    y_pred_gini = clf_gini.predict(X_test)
    
    clf_entropy = DecisionTreeClassifier(criterion="entropy", random_state=42)
    clf_entropy.fit(X_train, y_train)
    y_pred_entropy = clf_entropy.predict(X_test)
    
    evaluate_model(y_test, y_pred_gini, "Gini Criterion")
    evaluate_model(y_test, y_pred_entropy, "Entropy Criterion")


if __name__ == "__main__":
    main()