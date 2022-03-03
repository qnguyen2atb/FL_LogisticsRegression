import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegression():
    def __init__(self):
        self.name = 'LR'
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs):
        x = self._transform_x(x)
        y = self._transform_y(y)
        
