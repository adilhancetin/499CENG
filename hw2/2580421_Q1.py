from ucimlrepo import fetch_ucirepo
import os
import random
import math 
from sklearn.model_selection import train_test_split
        
FEATURES = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety']     
  
def fetch_dataset():
    car_evaluation = fetch_ucirepo(id=19) 
  
    X = car_evaluation.data.features 
    y = car_evaluation.data.targets 
    
    y = y['class'].apply(lambda x: 1 if x == 'unacc' else -1).values

    # Using the exact splitting snippet requested
    X_train , X_test , y_train , y_test = train_test_split (
        X , y , test_size =0.2 , random_state=42, stratify=y
    )
    return X_train , X_test , y_train , y_test

def encode_features(car_train):
    categories = {
        'buying': ['vhigh', 'high', 'med', 'low'],
        'maint': ['vhigh', 'high', 'med', 'low'],
        'doors': ['2', '3', '4', '5more'],
        'persons': ['2', '4', 'more'],
        'lug_boot': ['small', 'med', 'big'],
        'safety': ['low', 'med', 'high']
    }
    
    encoded_car_features = []
    for _, cars in car_train.iterrows():
        one_hot_encode = []
        for feature in FEATURES:
            val = cars[feature]
            for category in categories[feature]:
                if val == category:
                    one_hot_encode.append(1)
                else:
                    one_hot_encode.append(0)
        encoded_car_features.append(one_hot_encode)
    return encoded_car_features

def training_loop(train_feature, train_category, epochs=10, learning_rate=1.0):
    weights = [0.0] * len(train_feature[0])
    bias = 0.0
    
    for epoch in range(epochs):
        for i in range(len(train_feature)):
            activation = bias + sum(w * x for w, x in zip(weights, train_feature[i]))
            prediction = 1 if activation >= 0 else -1
            
            if prediction != train_category[i]:
                bias += learning_rate * train_category[i]
                for j in range(len(weights)):
                    weights[j] += learning_rate * train_category[i] * train_feature[i][j]
                    
    return weights, bias

def evaluate(test_feature, train_test, weights, bias):
    tp, fp, tn, fn = 0, 0, 0, 0
    
    for i in range(len(test_feature)):
        activation = bias + sum(w * x for w, x in zip(weights, test_feature[i]))
        prediction = 1 if activation >= 0 else -1
        
        if prediction == 1 and train_test[i] == 1:
            tp += 1
        elif prediction == 1 and train_test[i] == -1:
            fp += 1
        elif prediction == -1 and train_test[i] == -1:
            tn += 1
        elif prediction == -1 and train_test[i] == 1:
            fn += 1
            
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return accuracy, precision, recall, f1

def run_experiment(epochs, train_feature, train_category, test_feature, train_test):
    weights, bias = training_loop(train_feature, train_category, epochs=epochs, learning_rate=1.0)
    acc, prec, rec, f1 = evaluate(test_feature, train_test, weights, bias)
    
    print(f"--- Results for {epochs} epochs ---")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall   : {rec:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print()

def main():
    train_feature , test_feature , train_category , train_test = fetch_dataset() 
    
    encoded_car_features = encode_features(train_feature)
    encoded_test_features = encode_features(test_feature)
    
    run_experiment(10, encoded_car_features, train_category, encoded_test_features, train_test)
    run_experiment(20, encoded_car_features, train_category, encoded_test_features, train_test)

if __name__ == "__main__":
    main()    
    