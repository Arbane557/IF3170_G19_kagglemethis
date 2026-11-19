import numpy as np

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False

class DTL:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_feature = max_features
        self.random_state = random_state
        self.classes = None
        self.root = None

    # helper
    def entropy (self, y):
        if len(y) == 0:
            return 0

        count = np.bincount(y)
        p = count[count > 0] / len(y)
        return -np.sum(p * np.log2(p))

    def gain(self, y, y_left, y_right):
        n = len(y)
        nl = len(y_left)
        nr = len(y_right)

        if nl == 0 or nr == 0:
            return 0

        parent_entropy = self.entropy(y)
        weighted_entropy = (nl / n) * self.entropy(y_left) + (nr / n) * self.entropy(y_right)

        return parent_entropy - weighted_entropy

    def split(self, X, y):
        n, k = X.shape

        best = -1
        best_feature = None
        best_threshold = None
        best_left = None
        best_right = None

        for feature in range(k):
            val = np.unique(X[:, feature])
            if len(val) < 2:
                continue

            for i in range(len(val) - 1):
                threshold = (val[i] + val[i + 1]) / 2

                left = X[:, feature] <= threshold
                right = X[:, feature] > threshold

                if sum(left) < self.min_samples_leaf or sum(right) < self.min_samples_leaf:
                    continue

                gain = self.gain(y, y[left], y[right])

                if gain > best:
                    best = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = np.where(left)[0]
                    best_right = np.where(right)[0]

        return best_feature, best_threshold, best_left, best_right
    
    def tree(self, X, y, depth):
        new = Node()

        if len(np.unique(y)) == 1 or len(y) < self.min_samples_split or (self.max_depth is not None and depth >= self.max_depth):
            new.is_leaf = True
            new.value = np.bincount(y).argmax()
            return new

        feature, threshold, left, right = self.split(X, y)

        if feature is None:
            new.is_leaf = True
            new.value = np.bincount(y).argmax()
            return new

        new.feature = feature
        new.threshold = threshold
        
        new.left = self.tree(X[left], y[left], depth + 1)
        new.right = self.tree(X[right], y[right], depth + 1)    

        return new

    def fit(self, X, y):
        x = np.asarray(X)
        y = np.asarray(y)
        self.classes, y = np.unique(y, return_inverse=True)
        self.root = self.tree(x, y, depth=0)
        return self

    def predict(self, X):
        X = np.asarray(X)
        predictions = []
        for x in X:
            node = self.root
            while not node.is_leaf:
                if x[node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            predictions.append(self.classes[node.value])
        return self.classes[np.array(predictions)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y).ravel()

        return np.mean(predictions == y)