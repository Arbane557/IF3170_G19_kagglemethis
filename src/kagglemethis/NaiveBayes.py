import numpy as np
from collections import Counter


class NaiveBayes:
    def __init__(self, var_smoothing=1e-9, alpha=1.0,):
        self.var_smoothing = var_smoothing
        self.alpha = alpha
        self.classes = np.array([])

    def fit(self, X, y,  categorical_features=None):
        """
        categorical_features=None artinya semua feature dianggep numerik
        categorical_features="all" artinya semua feature dianggep kategorikal

        kalo mau mixed, misal data X = [123, 'Ngawi', 5000, 'STEIK'], maka 
        categorical_features = [1,3] (daerah dan fakultas)
        """

        if X is None or y is None or len(X) == 0 or len(y) == 0:
            return self

        X_arr = np.asarray(X, dtype=object)
        y_arr = np.asarray(y)
        n_samples, n_features = X_arr.shape

        self.classes = np.unique(y_arr)
        n_classes = len(self.classes)
        self.log_priors = np.zeros(n_classes, dtype=np.float64)

        if categorical_features == "all":
            self.cat_cols = list(range(n_features))
        elif categorical_features is None:
            self.cat_cols = []
        else:
            self.cat_cols = list(categorical_features)

        self.num_cols = [i for i in range(n_features) if i not in self.cat_cols]

        for idx, c in enumerate(self.classes):
            self.log_priors[idx] = np.log(np.sum(y_arr == c) / n_samples)

        if self.num_cols:
            X_num = X_arr[:, self.num_cols].astype(np.float64)
            means = np.zeros((n_classes, len(self.num_cols)), dtype=np.float64)
            variances = np.zeros((n_classes, len(self.num_cols)), dtype=np.float64)

            for idx, c in enumerate(self.classes):
                X_c = X_num[y_arr == c]
                means[idx, :] = np.mean(X_c, axis=0)
                variances[idx, :] = np.var(X_c, axis=0)

            epsilon = self.var_smoothing * np.var(X_num, axis=0).max()
            variances += epsilon

            self.means = means
            self.variances = variances

        if self.cat_cols:
            self.cat_log_probs = [{} for _ in range(n_classes)]
            self.cat_unseen_log_probs = [{} for _ in range(n_classes)]

            for col_idx in self.cat_cols:
                X_col = X_arr[:, col_idx]
                unique_cats = set(X_col)
                n_categories = len(unique_cats)

                for idx, c in enumerate(self.classes):
                    X_c_col = X_col[y_arr == c]
                    counts = Counter(X_c_col)
                    denom = len(X_c_col) + self.alpha * n_categories

                    self.cat_log_probs[idx][col_idx] = {
                        cat: np.log((counts[cat] + self.alpha) / denom)
                        for cat in unique_cats
                    }
                    self.cat_unseen_log_probs[idx][col_idx] = (
                        np.log(self.alpha / denom) if self.alpha > 0 else -np.inf
                    )

        return self

    def _joint_log_likelihood(self, X):
        X_arr = np.asarray(X, dtype=object)
        n_samples = X_arr.shape[0]
        n_classes = len(self.classes)

        joint_log_prob = np.tile(self.log_priors, (n_samples, 1))

        if self.num_cols:
            X_num = X_arr[:, self.num_cols].astype(np.float64)
            for idx in range(n_classes):
                mean = self.means[idx]
                var = self.variances[idx]
                log_gauss = -0.5 * np.sum(np.log(2.0 * np.pi * var)) - 0.5 * np.sum(
                    ((X_num - mean) ** 2) / var, axis=1
                )
                joint_log_prob[:, idx] += log_gauss

        if self.cat_cols:
            for idx in range(n_classes):
                for col_idx in self.cat_cols:
                    prob_map = self.cat_log_probs[idx][col_idx]
                    unseen_prob = self.cat_unseen_log_probs[idx][col_idx]
                    joint_log_prob[:, idx] += np.array(
                        [prob_map.get(val, unseen_prob) for val in X_arr[:, col_idx]],
                        dtype=np.float64,
                    )

        return joint_log_prob

    def predict(self, X):
        if len(self.classes) == 0:
            return None
        joint_log_prob = self._joint_log_likelihood(X)
        return self.classes[np.argmax(joint_log_prob, axis=1)]

    def predict_proba(self, X):
        if len(self.classes) == 0:
            return None
        joint_log_prob = self._joint_log_likelihood(X)
        log_prob_shifted = joint_log_prob - np.max(joint_log_prob, axis=1, keepdims=True)
        probs = np.exp(log_prob_shifted)
        return probs / np.sum(probs, axis=1, keepdims=True)
    
    def score(self, X, y):
        if len(self.classes) == 0 or X is None or y is None or len(y) == 0:
            return None
        y_pred = self.predict(X)
        y_arr = np.asarray(y)
        return float(np.mean(y_pred == y_arr))