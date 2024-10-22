import torch
from sklearn import svm
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=128, num_layers=2):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x
    
class SVMClassifier:
    def __init__(self, input_dim, num_classes=10):
        self.model = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
        self.input_dim = input_dim
        self.num_classes = num_classes

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
    
class LogisticRegression:
    def __init__(self, input_dim, num_classes=10):
        self.model = nn.Linear(input_dim, num_classes)
        self.input_dim = input_dim
        self.num_classes = num_classes

    def fit(self, X, y):
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        for epoch in range(100):
            optimizer.zero_grad()
            output = self.model(X)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        return self.model(X)

