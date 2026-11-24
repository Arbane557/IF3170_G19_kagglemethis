import numpy as np

from .bonus.logreg_dashboard import record_training, training_recorder


class LogisticRegression:
    def __init__(self, rate=0.01, epochs=300, penalty=None, strength=0.01, ratio=0.5, random_state=0, record=False, dashboard_path=None):
        self.rate = rate
        self.epochs = epochs
        self.penalty = penalty
        self.strength = strength
        self.ratio = ratio
        self.random_state = random_state
        self.dashboard_path = dashboard_path
        self.record = record

        self.weights = None
        self.bias = 0.0
        self.classes = None

    @record_training
    def fit(self, data, target):
        # Each row contain one sample and each column contain one numeric feature
        # Cleaning and scaling should already be done before training
        data = np.asarray(data, dtype=float)
        target = np.asarray(target)

        classes = np.unique(target)

        # Convert second sorted class into one and first class into zero
        # Keep original labels so prediction can return the same label type
        self.classes = classes
        labels = (target == self.classes[1]).astype(float)

        self.weights = np.zeros(data.shape[1])
        self.bias = 0.0
        rng = np.random.default_rng(self.random_state)
        recorder = training_recorder(self)
        if recorder is not None:
            recorder.observe(self, 0)

        for epoch in range(self.epochs):
            first_step = epoch * len(data) + 1
            for step, index in enumerate(rng.permutation(len(data)), start=first_step):
                sample = data[index]
                score = sample @ self.weights + self.bias
                probability = np.exp(-np.logaddexp(0, -score))

                # Ascend the log-likelihood after each individual sample
                # https://cs229.stanford.edu/summer2023/cs229-notes1.pdf#page=16
                error = labels[index] - probability
                gradient = error * sample

                # A sample gradient estimates the mean log-likelihood gradient.
                # Subtract the penalty gradient to maximize that penalized mean
                if self.penalty == "l1":
                    gradient -= self.strength * np.sign(self.weights)

                elif self.penalty == "l2":
                    gradient -= self.strength * self.weights

                elif self.penalty == "elasticnet":
                    gradient -= self.strength * (self.ratio * np.sign(self.weights) + (1 - self.ratio) * self.weights)

                self.weights += self.rate * gradient
                self.bias += self.rate * error

                if recorder is not None:
                    recorder.observe(self, step)

        return self

    def predict_proba(self, data):
        data = np.asarray(data, dtype=float)

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
