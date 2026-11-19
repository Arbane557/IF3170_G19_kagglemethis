import numpy as np


class LogisticRegression:
    def __init__(self, rate=0.01, epochs=1000, penalty=None, strength=0.01, ratio=0.5):
        # Rate control how far weights move in every update
        # Strength control penalty size and ratio control L1 share in Elastic Net
        if not np.isfinite(rate) or rate <= 0:
            raise ValueError("Rate must be positive and finite")

        if not isinstance(epochs, int) or epochs <= 0:
            raise ValueError("Epochs must be a positive integer")

        if penalty not in [None, "l1", "l2", "elasticnet"]:
            raise ValueError("Penalty must be None, l1, l2, or elasticnet")

        if not np.isfinite(strength) or strength < 0:
            raise ValueError("Strength must be nonnegative and finite")

        if not np.isfinite(ratio) or ratio < 0 or ratio > 1:
            raise ValueError("Ratio must be between zero and one")

        self.rate = rate
        self.epochs = epochs
        self.penalty = penalty
        self.strength = strength
        self.ratio = ratio

        self.weights = None
        self.bias = 0.0
        self.classes = None

    def fit(self, data, target):
        # Each row contain one sample and each column contain one numeric feature
        # Cleaning and scaling should already be done before training
        data = np.asarray(data, dtype=float)
        target = np.asarray(target)

        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] == 0:
            raise ValueError("Data must be a nonempty matrix")

        if target.ndim != 1 or len(target) != len(data):
            raise ValueError("Target must contain one label for each sample")

        if not np.all(np.isfinite(data)):
            raise ValueError("Data must contain only finite numbers")

        classes = np.unique(target)

        if len(classes) != 2:
            raise ValueError("Logistic Regression needs exactly two classes")

        # Convert second sorted class into one and first class into zero
        # Keep original labels so prediction can return the same label type
        self.classes = classes
        labels = (target == self.classes[1]).astype(float)

        self.weights = np.zeros(data.shape[1])
        self.bias = 0.0

        for epoch in range(self.epochs):
            # Sigmoid turn linear score into positive class probability
            # Logaddexp keep the same sigmoid result without exponential overflow
            scores = data @ self.weights + self.bias
            probability = np.exp(-np.logaddexp(0, -scores))

            # Mean log loss give this gradient for all training samples
            # Bias gradient only need the average prediction error
            error = probability - labels
            gradient = data.T @ error / len(data)
            bias = np.mean(error)

            # L1 use absolute weight sum and L2 use half squared weight sum
            # Elastic Net mix both penalties using ratio as the L1 share
            if self.penalty == "l1":
                gradient += self.strength * np.sign(self.weights)

            elif self.penalty == "l2":
                gradient += self.strength * self.weights

            elif self.penalty == "elasticnet":
                gradient += self.strength * (
                    self.ratio * np.sign(self.weights)
                    + (1 - self.ratio) * self.weights
                )

            # Move weights and bias against the gradient to reduce loss
            # Penalty only affect weights so bias is never regularized
            self.weights -= self.rate * gradient
            self.bias -= self.rate * bias

        return self

    def predict_proba(self, data):
        # Prediction need fitted weights and the same number of feature columns
        # Keep the input as a matrix even when predicting only one sample
        if self.weights is None:
            raise ValueError("Fit the model before prediction")

        data = np.asarray(data, dtype=float)

        if data.ndim != 2 or data.shape[1] != len(self.weights):
            raise ValueError("Data must have the same feature count as training")

        if not np.all(np.isfinite(data)):
            raise ValueError("Data must contain only finite numbers")

        scores = data @ self.weights + self.bias
        probability = np.exp(-np.logaddexp(0, -scores))

        # Columns follow the sorted class order stored during training
        # Negative class probability is one minus positive class probability
        return np.column_stack((1 - probability, probability))

    def predict(self, data):
        # Probability at least half select the second sorted class
        # Return original labels instead of the internal zero and one encoding
        probability = self.predict_proba(data)[:, 1]
        labels = (probability >= 0.5).astype(int)

        return self.classes[labels]

    def score(self, data, target):
        predictions = self.predict(data)
        return np.mean(predictions == np.asarray(target))