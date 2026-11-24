import numpy as np
import matplotlib.pyplot as plt

class Node:
    def __init__(self):
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.value = None
        self.is_leaf = False

class DTL:
    def __init__(self, criterion='entropy', max_depth=None, min_samples_split=2, min_samples_leaf=1, max_features=None, random_state=None,  min_impurity_decrease = 0.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_feature = max_features
        self.random_state = random_state
        self.classes = None
        self.root = None
        self.min_impurity_decrease = min_impurity_decrease

    # helper
    def entropy (self, y):
        if len(y) == 0:
            return 0

        count = np.bincount(y)
        p = count[count > 0] / len(y)
        return -np.sum(p * np.log2(p))
    
    def gini(self, y):
        if len(y) == 0:
            return 0
        count = np.bincount(y)
        p = count[count > 0] / len(y)
        return 1 - np.sum(p ** 2)

    def impurity(self, y):
        return self.entropy(y) if self.criterion == 'entropy' else self.gini(y)
    

    def gain(self, y, y_left, y_right):
        n = len(y)
        nl = len(y_left)
        nr = len(y_right)

        if nl == 0 or nr == 0:
            return 0

        parent_entropy = self.impurity(y)
        weighted_entropy = (nl / n) * self.impurity(y_left) + (nr / n) * self.impurity(y_right)

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

                if np.count_nonzero(left) < self.min_samples_leaf or np.count_nonzero(right) < self.min_samples_leaf:
                    continue

                gain = self.gain(y, y[left], y[right])

                if gain >= self.min_impurity_decrease and gain > best:
                    best = gain
                    best_feature = feature
                    best_threshold = threshold
                    best_left = np.where(left)[0]
                    best_right = np.where(right)[0]

        return best_feature, best_threshold, best_left, best_right
    
    def viz(self, top_n=3, feature_names=None, class_names=None):
        from collections import deque

        fig, ax = plt.subplots(figsize=(20, 10))
        ax.axis('off')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        labels = self.classes if class_names is None else class_names
        queue = deque([(self.root, 0, 0.0, 1.0, None)])

        while queue:
            node, depth, lx, rx, parent_xy = queue.popleft()
            if depth > top_n:
                continue

            cx = (lx + rx) / 2
            cy = 1.0 - (depth + 0.5) / (top_n + 1)

            if node.is_leaf or depth == top_n:
                if node.is_leaf:
                    v = node.value
                else:
                    fname = (
                        feature_names[node.feature] if feature_names is not None
                        else f"X[{node.feature}]"
                    )
                    label = f"[{fname}\n≤ {node.threshold:.2f}]"
                    color = '#fffacd'
                    ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                            bbox=dict(boxstyle='round', facecolor=color, edgecolor='gray'),
                            wrap=True)
                    if parent_xy:
                        ax.annotate('', xy=(cx, cy + 0.03), xytext=parent_xy,
                                    arrowprops=dict(arrowstyle='->', color='gray'))
                    continue

                label = str(labels[v])
                color = '#ff9999' if label == 'canceled' else '#90ee90'
            else:
                fname = (
                    feature_names[node.feature] if feature_names is not None
                    else f"X[{node.feature}]"
                )
                label = f"{fname}\n≤ {node.threshold:.2f}"
                color = '#add8e6'

            ax.text(cx, cy, label, ha='center', va='center', fontsize=7,
                    bbox=dict(boxstyle='round', facecolor=color, edgecolor='gray'),
                    wrap=True)

            if parent_xy:
                ax.annotate('', xy=(cx, cy + 0.03), xytext=parent_xy,
                            arrowprops=dict(arrowstyle='->', color='gray'))

            if not node.is_leaf and depth < top_n:
                parent = (cx, cy - 0.03)
                queue.append((node.left, depth + 1, lx, cx, parent))
                queue.append((node.right, depth + 1, cx, rx, parent))

        return fig

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
            predictions.append(node.value)
        return self.classes[np.array(predictions)]

    def score(self, X, y):
        predictions = self.predict(X)
        y = np.asarray(y).ravel()

        return np.mean(predictions == y)
