import numpy as np

class KNN:
    def __init__(self, k=3, metric="euclidean", p=2):
        if k < 1:
            raise ValueError("K harus minimal 1")

        valid_metrics = [
            "euclidean",
            "manhattan",
            "minkowski",
        ]

        if metric not in valid_metrics:
            raise ValueError("Metric tidak dikenal")

        self.k = k
        self.metric = metric
        self.p = p
        self.X_train = None
        self.y_train = None

    def fit(self, x, y):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y)

        if x.ndim != 2 :
            raise ValueError("X harus berupa matrix 2D")
        if len(x) != len(y) :
            raise ValueError("Jumlah X dan Y harus sama")
        if self.k > len(x):
            raise ValueError("Jumlah k melebehih jumlah baris")
        if not np.isfinite(x).all():
            raise ValueError("X mengandung NaN atau infinity")

        self.X_train = x
        self.y_train = y
        return self

    def calculate_distance(self, x1, x2):
        if self.metric == "euclidean":
            return np.sqrt(np.sum((x1 - x2) ** 2))
        
        elif self.metric == "manhattan":
            return np.sum(np.abs(x1 - x2))

        elif self.metric == "minkowski":
            return np.sum(
                np.abs(x1 - x2) ** self.p
            ) ** (1 / self.p)

        raise ValueError(f"Metric tidak dikenal: {self.metric}")

    def predict(self, x):
        if self.X_train is None:
            raise ValueError("Model Belum dilatih")

        x = np.asarray(x, dtype=float)

        if x.ndim == 1:
            x = x.reshape(1, -1)

        predictions = []

        for rows in x:
            distances = [self.calculate_distance(rows, x_train) for x_train in self.X_train]

            nearest_indices = np.argsort(distances, kind="stable")[:self.k]
            nearest_labels = self.y_train[nearest_indices]

            labels, counts = np.unique(nearest_labels, return_counts=True)
            prediction = labels[np.argmax(counts)]
            predictions.append(prediction)

        return np.array(predictions)

    def score(self, x, y):
        predictions = self.predict(x)
        return float(np.mean(predictions == np.asarray(y)))

