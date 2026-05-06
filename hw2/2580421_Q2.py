import torch
import torch.nn as nn
import torch.optim as optim
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def prepare_dataset():
    dry_bean = fetch_ucirepo(id=602)
    
    X = dry_bean.data.features 
    y = dry_bean.data.targets 

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y.values.ravel())

    X_train_dev , X_test , y_train_dev , y_test = train_test_split (
        X , y , test_size =0.2 , random_state=42, stratify=y
    )
    X_train , X_dev , y_train , y_dev = train_test_split (
        X_train_dev , y_train_dev , test_size =0.125 , random_state=42, stratify=y_train_dev
    )

    X_train = torch.tensor(X_train, dtype=torch.float32)
    X_dev = torch.tensor(X_dev, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.float32) 

    y_train = torch.tensor(y_train, dtype=torch.long)
    y_dev = torch.tensor(y_dev, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.long)

    return X_train, X_dev, X_test, y_train, y_dev, y_test

class ANN_1(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN_1, self).__init__()
        self.hidden = nn.Linear(input_dim, 16)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

class ANN_2(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN_2, self).__init__()
        self.hidden1 = nn.Linear(input_dim, 16)
        self.sigmoid1 = nn.Sigmoid()
        self.hidden2 = nn.Linear(16, 16)
        self.sigmoid2 = nn.Sigmoid()
        self.output = nn.Linear(16, output_dim)
        
    def forward(self, x):
        x = self.sigmoid1(self.hidden1(x))
        x = self.sigmoid2(self.hidden2(x))
        x = self.output(x)
        return x

class ANN_3(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ANN_3, self).__init__()
        self.hidden = nn.Linear(input_dim, 32)
        self.sigmoid = nn.Sigmoid()
        self.output = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = self.sigmoid(self.hidden(x))
        x = self.output(x)
        return x

def train_and_evaluate(model, X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=200, lr=0.01):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()         
        optimizer.zero_grad() 

        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        dev_outputs = model(X_dev)
        _, dev_predictions = torch.max(dev_outputs, 1)
        
        y_dev_np = y_dev.numpy()
        dev_preds_np = dev_predictions.numpy()
        
        dev_acc = accuracy_score(y_dev_np, dev_preds_np)
        dev_prec = precision_score(y_dev_np, dev_preds_np, average='macro', zero_division=0)
        dev_rec = recall_score(y_dev_np, dev_preds_np, average='macro', zero_division=0)
        dev_f1 = f1_score(y_dev_np, dev_preds_np, average='macro', zero_division=0)

        test_outputs = model(X_test)
        _, test_predictions = torch.max(test_outputs, 1)
        
        y_test_np = y_test.numpy()
        test_preds_np = test_predictions.numpy()
        
        test_acc = accuracy_score(y_test_np, test_preds_np)
        test_prec = precision_score(y_test_np, test_preds_np, average='macro', zero_division=0)
        test_rec = recall_score(y_test_np, test_preds_np, average='macro', zero_division=0)
        test_f1 = f1_score(y_test_np, test_preds_np, average='macro', zero_division=0)

    dev_metrics = {'acc': dev_acc, 'prec': dev_prec, 'rec': dev_rec, 'f1': dev_f1}
    test_metrics = {'acc': test_acc, 'prec': test_prec, 'rec': test_rec, 'f1': test_f1}
    return dev_metrics, test_metrics

def main():
    X_train, X_dev, X_test, y_train, y_dev, y_test = prepare_dataset()
    
    input_dim = X_train.shape[1]
    output_dim = len(torch.unique(y_train))

    print("Training and Evaluating Models...\n")
    
    print("--- 1 Hidden Layer (16 neurons) ---")
    model_1 = ANN_1(input_dim, output_dim)
    dev_metrics_1, test_metrics_1 = train_and_evaluate(model_1, X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=500, lr=0.01)
    
    print("DEVELOPMENT SET:")
    print("Accuracy: ", dev_metrics_1['acc'])
    print("Precision:", dev_metrics_1['prec'])
    print("Recall:   ", dev_metrics_1['rec'])
    print("F1:       ", dev_metrics_1['f1'])
    print("TEST SET:")
    print("Accuracy: ", test_metrics_1['acc'])
    print("Precision:", test_metrics_1['prec'])
    print("Recall:   ", test_metrics_1['rec'])
    print("F1:       ", test_metrics_1['f1'])
    print()


    print("--- 2 Hidden Layers (16-16 neurons) ---")
    model_2 = ANN_2(input_dim, output_dim)
    dev_metrics_2, test_metrics_2 = train_and_evaluate(model_2, X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=500, lr=0.01)
    
    print("DEVELOPMENT SET:")
    print("Accuracy: ", dev_metrics_2['acc'])
    print("Precision:", dev_metrics_2['prec'])
    print("Recall:   ", dev_metrics_2['rec'])
    print("F1:       ", dev_metrics_2['f1'])
    print("TEST SET:")
    print("Accuracy: ", test_metrics_2['acc'])
    print("Precision:", test_metrics_2['prec'])
    print("Recall:   ", test_metrics_2['rec'])
    print("F1:       ", test_metrics_2['f1'])
    print()


    print("--- 1 Hidden Layer (32 neurons) ---")
    model_3 = ANN_3(input_dim, output_dim)
    dev_metrics_3, test_metrics_3 = train_and_evaluate(model_3, X_train, y_train, X_dev, y_dev, X_test, y_test, epochs=500, lr=0.01)
    
    print("DEVELOPMENT SET:")
    print("Accuracy: ", dev_metrics_3['acc'])
    print("Precision:", dev_metrics_3['prec'])
    print("Recall:   ", dev_metrics_3['rec'])
    print("F1:       ", dev_metrics_3['f1'])
    print("TEST SET:")
    print("Accuracy: ", test_metrics_3['acc'])
    print("Precision:", test_metrics_3['prec'])
    print("Recall:   ", test_metrics_3['rec'])
    print("F1:       ", test_metrics_3['f1'])
    print()

if __name__ == "__main__":
    main()