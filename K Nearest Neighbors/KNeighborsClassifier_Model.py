import numpy as np
import sqlite3


class MyKNeighborsClassifier:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        if n_neighbors <= 0:
            raise ValueError("n_neighbors value must be greater than 0")
        self.n_neighbors = n_neighbors
        
        if metric not in ['euclidean','Manhattan','cosine']:
            raise ValueError("the metric value must be one of euclidean, cosine or Manhattan")
        self.metric = metric
        
    def fit(self, x, y):
        self.X_train = x.values
        self.Y_train = y.values
    
    
    def predict(self, X_test):
        predictions = []
        for x in X_test.values:
            predictions.append(self._predict(x))
        return predictions
    
    def euclidean_distance(self, x, x_train):
        return np.sqrt(np.sum((x_train - x)**2))
    def Manhattan_distance(self, x, x_train):
        return np.sum(np.abs(x_train - x))
    def cosine_similarity(self, x, x_train):
        return np.dot(x, x_train) / (np.linalg.norm(x) * np.linalg.norm(x_train))
    
    def _predict(self, point):
        bit = 1
        distances = []
        # Finding distances
        if self.metric == 'euclidean':
            distances = np.array([self.euclidean_distance(point,x_train) for x_train in self.X_train])
        elif self.metric == 'Manhattan':
            distances = np.array([self.Manhattan_distance(point,x_train) for x_train in self.X_train])
        else:
            distances = np.array([self.cosine_similarity(point,x_train) for x_train in self.X_train])
            bit = -1
        
        # find the first k indices of minimum distances or maximum similarities
        k_indices = distances.argsort()[::bit][:self.n_neighbors]
        
        # return majority of k distances
        k_classes_labels, classes_counts = np.unique(self.Y_train[k_indices], return_counts=True)
        return k_classes_labels[np.argmax(classes_counts)] 
    
    
    
    
    

    
    
