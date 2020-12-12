import pandas as pd
import random, math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial import distance


# https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X = pd.DataFrame(X, columns=X.columns.tolist())
X.head()

def euclid(row1, row2):
    if len(row1) != len(row2):
        print('bad length')
        return None
    
    rolling = 0.0
    for i in range(len(row1)):
        rolling += (row1[i] - row2[i])**2
    return math.sqrt(rolling)

def knn(data, k=3):
    x_new = scaler.transform(pd.DataFrame(data).T)[0]

    # compute euclidean distances
    dists = np.empty(len(X))
    for i, row in X.iterrows():
        dists[i] = euclid(x_new, row)

    # get top k indices
    knn_indices = pd.Series(dists).sort_values()[:k].index.tolist()

    if y[knn_indices].mean() <= .5:
        pred = 0
    else:
        pred = 1

    return pred


def valueCounts(vector, index):
    try:
        return vector.value_counts()[index]
    except:
        return 0
    
# H(x)
def h(x):
    lenx = len(x)
    len0 = valueCounts(x, 0)
    len1 = valueCounts(x, 1)
    try:
        i0 = len0 * math.log(lenx/len0, 2) / lenx
    except:
        i0 = 0
    try:
        i1 = len1 * math.log(lenx/len1, 2) / lenx
    except:
        i1 = 0
    return i0 + i1

# H(x | y)
def hc(x, y):
    leny = len(y)
    rolling = 0
    
    for xval in [0,1]:
        for yval in [0,1]:
            p_xy = len(y[x==xval][y==yval]) / len(x)
            if p_xy != 0:
                rolling += p_xy * math.log(valueCounts(y, yval) / leny / p_xy, 2)
    return rolling
  
# I(x,y)
def I(x, y):
    return h(x) - hc(x, y)

class Node():
    def __init__(self):
        self.left = None
        self.right = None
        self.i = None
        self.feature = None
        self.samples = None
        self.survived = None
        self.info = None

class DTree():
    def __init__(self, X, y):
        self.root = Node()
        buildTree(self.root, X, y, min_leaf=int(len(y)*0.05))
    
    def predict(self, row):
        return self._predict(self.root, row)
    
    def _predict(self, node, row):
        if node.left is None:
            return node.survived
        
        if row[node.feature] == 0:
            return self._predict(node.left, row)
        else:
            return self._predict(node.right, row)
        
    def display(self):
        self._display(self.root, 0)
        
    def _display(self, node, depth):
        
        if node.left is not None:
            self._display(node.left, depth+1)
        
        if node.left is not None:
            print(" "*(3*depth), depth, node.feature, "class: {}".format(node.survived))
        else:
            print(" "*(3*depth), depth, "class: {} with {} rows --".format(node.survived, node.samples), node.info)
        
        if node.right is not None:
            self._display(node.right, depth+1)
            
def buildTree(node, X, y, min_leaf=3):
    # stop - min samples
    if len(y) < min_leaf:
        node.survived = 0 if valueCounts(y, 0) >= valueCounts(y, 1) else 1
        node.samples = len(y)
        node.info = "hit min_leaf"
        return
    
    # stop - all ys have same value
    if len(y.value_counts()) < 2:
        node.survived = 0 if valueCounts(y, 0) >= valueCounts(y, 1) else 1
        node.samples = len(y)
        node.info = "same y value"
        return
        
    Is = {x: I(X[x], y) for x in X}
    
    node.feature = max(Is, key=Is.get)
    node.i = max(Is.values()) 
    
    # stop - too little information
    if float(node.i) < 0.015:
        node.survived = 0 if valueCounts(y, 0) >= valueCounts(y, 1) else 1
        node.samples = len(y)
        node.info = "small max I(x,y)"
        return
    
    node.survived = 0 if valueCounts(y, 0) >= valueCounts(y, 1) else 1 # 1 gets ties
    node.samples = len(y)
        
    node.left = Node()
    buildTree(node.left, X[X[node.feature] == 0], y[X[node.feature] == 0], min_leaf)

    node.right = Node()
    buildTree(node.right, X[X[node.feature] == 1], y[X[node.feature] == 1], min_leaf)
    
    return

class RandomForest():
    def __init__(self, X, y, mode='portion', num=1):
        self.trees = []
        
        if mode == 'portion':
            self.portion(X, y, num)
        
        if mode == 'dropc':
            self.dropColumn(X, y, num)
            
    def portion(self, X, y, num):
        for i in range(num):
            r_idx = random.sample(X.index.tolist(), k=int(len(X)*.8))
            X_r = X[X.index.isin(r_idx)]
            y_r = y[y.index.isin(r_idx)]
            
            tree = DTree(X_r, y_r)
            self.trees.append(tree)
    
    def dropColumn(self, X, y, num):
        for i in range(num):
            X_c = X.drop([X.columns[i]], axis=1)
            
            tree = DTree(X_c, y)
            self.trees.append(tree)
    
    def predict(self, row):
        results = []
        for tree in self.trees:
            results.append(tree.predict(row))
            
        return 0 if valueCounts(pd.Series(results), 0) >= valueCounts(pd.Series(results), 1) else 1