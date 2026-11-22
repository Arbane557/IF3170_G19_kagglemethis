import numpy as np
import matplotlib.pyplot as plt

# Uses the sequential minimal optimization (SMO) algorithm to compute the hyperplane of SVM model
# References: 
# - https://en.wikipedia.org/wiki/Sequential_minimal_optimization
# - https://cs229.stanford.edu/materials/smo.pdf (most code is taken from the pseudocode here)
# - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html (parameter and method name mostly match here)


class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=1000, random_state=None):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tolerance = tol
        self.max_iter = max_iter

        self._support = None
        self._support_vectors = None
        self._dual_coef = None
        self._intercept = 0.0
        self._classes = None
        self._gamma = None
        self._rng = np.random.default_rng(random_state)

        self._SMO_EPSILON = 1e-7

    # Linear kernel, just use the formula from slide
    @staticmethod
    def _kernel_linear(X1, X2):
        return np.dot(X1, X2.T)

    # Polynomial kernel, just use the formula from slide
    @staticmethod
    def _kernel_poly(X1, X2, gamma=1.0, coef0=1.0, degree=3):
        if gamma <= 0:
            raise ValueError(f"Invalid gamma value (gamma={gamma}) for polynomial kernel")

        if degree < 0:
            raise ValueError(f"Invalid degree value (degree={degree}) for polynomial kernel")

        return (gamma * np.dot(X1, X2.T) + coef0) ** degree

    # Gaussian kernel, just use the formula from slide
    @staticmethod
    def _kernel_gaussian(X1, X2, gamma=0.1):
        if gamma <= 0:
            raise ValueError(f"Invalid gamma value (gamma={gamma}) for Gaussian kernel")

        m = X1.shape[0]
        k = X2.shape[0]
        K = np.zeros((m, k))

        for i in range(m):
            for j in range(k):
                euclid_dist = np.sum((X1[i] - X2[j]) ** 2)
                K[i, j] = np.exp(-gamma * euclid_dist)

        return K
    
    # Sigmoid kernel, just use the formula from slide
    @staticmethod
    def _kernel_sigmoid(X1, X2, gamma=1.0, coef0=1.0):
        return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

    # Parameter validation and kernel computation for non-linear separable data
    def _compute_kernel(self, X1, X2):
        if self.kernel == 'linear':
            return self._kernel_linear(X1, X2)

        elif self.kernel == 'poly':
            return self._kernel_poly(X1, X2, gamma=self._gamma, coef0=self.coef0, degree=self.degree)

        elif self.kernel in ('rbf', 'gaussian'):
            return self._kernel_gaussian(X1, X2, gamma=self._gamma)

        elif self.kernel == 'sigmoid':
            return self._kernel_sigmoid(X1, X2, gamma=self._gamma, coef0=self.coef0)
    
        elif callable(self.kernel):
            return self.kernel(X1, X2)

        else:
            raise NotImplementedError(f"Unimplemented kernel: '{self.kernel}'. Choose 'linear', 'poly', 'gaussian'/'rbf', or provide a custom callable function as kernel")

    
    # Model fit function
    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Number of samples mismatch, X has {X.shape[0]} samples but y has {y.shape[0]} samples.")

        n_samples, n_features = X.shape

        # Map classification labels to -1 and +1 but store original for prediction later
        self._classes = np.unique(y)
        if len(self._classes) != 2:
            raise NotImplementedError(f"This SVC implementation does not support multi-class classification (number of class is {len(self._classes)})")
        y = np.where(y == self._classes[0], -1.0, 1.0)

        # Resolve gamma parameter (uses the same gamma calculation as scikit-learn gamma)
        if self.gamma == 'scale':
            var = X.var()
            self._gamma = 1.0
            if var > 0:
                self._gamma /= (n_features * var)

        elif self.gamma == 'auto':
            self._gamma = 1.0 / n_features

        elif isinstance(self.gamma, (float, int, np.number)) and self.gamma >= 0:
            self._gamma = float(self.gamma)

        else:
            raise NotImplementedError(f"Invalid gamma parameter passed: '{self.gamma}'. Please choose 'scale' or 'auto' or valid floating point number")


        # Some computed parameter for SMO algorithm
        K = self._compute_kernel(X, X)
        alphas = np.zeros(n_samples)
        b = 0.0
        passes = 0

        # General form of linear classifier: f(x) = w.T * x + b
        def f_(idx):
            return np.sum(alphas * K[:, idx] * y) + b

        # Main SMO solver, mostly is pseudocode from reference
        while passes < self.max_iter:
            num_changed_alphas = 0
            for i in range(n_samples):

                E_i = f_(i) - y[i]

                if (y[i] * E_i < -self.tolerance and alphas[i] < self.C) or (y[i] * E_i > self.tolerance and alphas[i] > 0):
                    
                    j = self._rng.choice([idx for idx in range(n_samples) if idx != i])
                    E_j = f_(j) - y[j]

                    alpha_i_old, alpha_j_old = alphas[i], alphas[j]

                    if y[i] != y[j]:
                        L = max(0.0, alpha_j_old - alpha_i_old)
                        H = min(self.C, self.C + alpha_j_old - alpha_i_old)
                    else:
                        L = max(0.0, alpha_i_old + alpha_j_old - self.C)
                        H = min(self.C, alpha_i_old + alpha_j_old)

                    if L == H:
                        continue

                    eta = 2.0 * K[i, j] - K[i, i] - K[j, j]
                    if eta >= 0:
                        continue

                    alphas[j] -= (y[j] * (E_i - E_j)) / eta
                    alphas[j] = np.clip(alphas[j], L, H)

                    if abs(alphas[j] - alpha_j_old) < self._SMO_EPSILON:
                        continue

                    alphas[i] += y[i] * y[j] * (alpha_j_old - alphas[j])

                    b1 = b - E_i - y[i] * (alphas[i] - alpha_i_old) * K[i, i] - y[j] * (alphas[j] - alpha_j_old) * K[i, j]
                    b2 = b - E_j - y[i] * (alphas[i] - alpha_i_old) * K[i, j] - y[j] * (alphas[j] - alpha_j_old) * K[j, j]

                    if 0 < alphas[i] < self.C:
                        b = b1
                    elif 0 < alphas[j] < self.C:
                        b = b2
                    else:
                        b = 0.5 * (b1 + b2)

                    num_changed_alphas += 1

            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0


        # Retrieve the result from SMO result for model prediction
        self._intercept = b
        self._support = []
        self._support_vectors = []
        self._dual_coef = []
        for i in range(len(alphas)):
            if alphas[i] > self._SMO_EPSILON:
                self._support.append(i)
                self._support_vectors.append(X[i])
                self._dual_coef.append(alphas[i] * y[i])

        if len(self._support) == 0:
            self._support = np.empty(0, dtype=int)
            self._support_vectors = np.empty((0, X.shape[1]))
            self._dual_coef = np.empty(0)
        else:
            self._support = np.array(self._support)
            self._support_vectors = np.array(self._support_vectors)
            self._dual_coef = np.array(self._dual_coef)

        return self

    # Evaluate decision function for the samples in X
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        return np.dot(self._compute_kernel(X, self._support_vectors), self._dual_coef) + self._intercept

    # Perform classification on samples in X
    def predict(self, X):
        X = np.asarray(X, dtype=float)
        return np.where(self.decision_function(X) >= 0, self._classes[1], self._classes[0])

    # Return accuracy (count of correct prediction) on provided data and labels
    def score(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).reshape(-1)
        if y.shape[0] != X.shape[0]:
            raise ValueError(f"Number of samples mismatch, X has {X.shape[0]} samples but y has {y.shape[0]} samples.")

        return np.mean(self.predict(X) == y)