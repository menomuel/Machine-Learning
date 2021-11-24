from typing import no_type_check_decorator
import matplotlib.pyplot as plt
import numpy as np
import random as rd
from sklearn.tree import DecisionTreeClassifier

def plot_mnist_sample(sample):
    two_d = (np.reshape(sample, (28, 28))).astype(np.uint8)
    plt.figure(figsize=(4, 4))
    plt.xticks([])
    plt.yticks([])
    plt.imshow(two_d, cmap=plt.cm.gray)


def plot_missclassified_samples(model, X, y, num=5):
    pred = model.predict(X)
    plt.figure(figsize=(20,4))
    for plotIndex, badIndex in enumerate(get_missclassified_ind(pred, y)[:num]):
        plt.subplot(1, num, plotIndex + 1)
        plt.imshow(np.reshape(X[badIndex], (28,28)), cmap=plt.cm.gray)
        plt.xticks([])
        plt.yticks([])
        plt.title("Predicted: {}, Actual: {}".format(pred[badIndex], y[badIndex]), fontsize = 15, color='w')

def get_missclassified_ind(pred, y):
    index = 0
    misclassifiedIndices = []
    for label, predict in zip(y, pred):
        if label != predict: 
            misclassifiedIndices.append(index)
        index +=1
    return misclassifiedIndices

class CustomRandomForestClassifier():
    def __init__(self, n_trees=10):
        self.n_trees_ = n_trees
        self.trees_ = [DecisionTreeClassifier(max_features='sqrt') for _ in range(n_trees)]

    def fit(self, X, y):
        for tree in self.trees_:
            ind_bagging = [rd.randint(0, X.shape[0] - 1) for _ in range(X.shape[0])]
            X_sub = X[ind_bagging]
            y_sub = y[ind_bagging]

            tree.fit(X_sub, y_sub)
    
    def predict(self, X):
        # Form table of each tree predictions
        table = np.zeros((X.shape[0], self.n_trees_), dtype=int)
        for i, tree in enumerate(self.trees_):
            pred = tree.predict(X)
            pred = [int(p) for p in pred]
            table[:, i] = pred
        
        # For each sample determine the most frequent prediction
        res = np.zeros(X.shape[0], dtype=int)
        for i in range(X.shape[0]):
            res[i] = np.argmax(np.bincount(table[i, :]))
        return res
    
    def score(self, X, y):
        y = [int(_y) for _y in y]
        mask = self.predict(X) == y
        pos_count = sum([True for el in mask if el > 0])
        return pos_count / len(mask)
