import numpy as np
import matplotlib.pyplot as plt

# Uses the sequential minimal optimization (SMO) algorithm to compute the hyperplane of SVM model
# References: 
# - https://en.wikipedia.org/wiki/Sequential_minimal_optimization
# - https://www.microsoft.com/en-us/research/wp-content/uploads/1998/04/sequential-minimal-optimization.pdf (most code is taken from the pseudocode here)
# - https://cs229.stanford.edu/materials/smo.pdf (standford pseudocode of SMO, a little correction is taken from here)
# - https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html (parameter and method name mostly match here)


class SVC:
    def __init__(self, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=1e-3, max_iter=1000, cache_size=2000, random_state=None):
        self._validate_param(kernel, gamma, degree)

        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.tolerance = tol
        self.max_cache_size = cache_size # Maximum cache size for kernel computation, in columns
        self.max_iter = max_iter

        self._coef = None
        self._support = None
        self._support_vectors = None
        self._dual_coef = None
        self._intercept = 0.0
        self._classes = None
        self._gamma = None
        self._rng = np.random.default_rng(random_state)

        self._SMO_EPSILON = 1e-7
        self._DECISION_BATCH_SIZE = 2000

    # Validation only done at class init to speedup
    def _validate_param(self, kernel, gamma, degree):
        if kernel == 'poly':
            if gamma != 'auto' and gamma != 'scale' and gamma <= 0:
                raise ValueError(f"Invalid gamma value (gamma={gamma}) for polynomial kernel")

            if degree < 0:
                raise ValueError(f"Invalid degree value (degree={degree}) for polynomial kernel")

        if kernel == 'rbf' or kernel == 'gaussian':
            if gamma != 'auto' and gamma != 'scale' and gamma <= 0:
                raise ValueError(f"Invalid gamma value (gamma={gamma}) for Gaussian kernel")

        if kernel == 'sigmoid':
            if gamma != 'auto' and gamma != 'scale' and gamma < 0:
                raise ValueError(f"Invalid gamma value (gamma={gamma}) for Sigmoid kernel")

    # Linear kernel, just use the formula from slide
    @staticmethod
    def _kernel_linear(X1, X2):
        return np.dot(X1, X2.T)

    # Polynomial kernel, just use the formula from slide
    @staticmethod
    def _kernel_poly(X1, X2, gamma=1.0, coef0=1.0, degree=3):
        return (gamma * np.dot(X1, X2.T) + coef0) ** degree


    # Gaussian kernel, just use the formula from slide
    @staticmethod
    def _kernel_gaussian(X1, X2, gamma=0.1):
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True).T
        sq_dists = np.maximum(X1_sq + X2_sq - 2.0 * np.dot(X1, X2.T), 0.0)
        return np.exp(-gamma * sq_dists)
    
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



    # Model fit function, uses the SMO algorithm
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
        alphas = np.zeros(n_samples)
        b = 0.0
        E = -y.copy()

        # The diagonal part of kernel matrix can be precomputed based on the kernel type
        if self.kernel in ('rbf', 'gaussian'):
            K_diag = np.ones(n_samples)
        elif self.kernel == 'linear':
            K_diag = np.sum(X * X, axis=1)
        elif self.kernel == 'poly':
            K_diag = (self._gamma * np.sum(X * X, axis=1) + self.coef0) ** self.degree
        else:
            K_diag = np.array([self._compute_kernel(X[i:i+1], X[i:i+1])[0, 0] for i in range(n_samples)])
        

        # Caching on kernel access so memory usage is acceptable
        is_gaussian = (self.kernel in ('rbf', 'gaussian'))
        X_square = None
        if is_gaussian:
            X_square = np.sum(X * X, axis=1)

        kernel_cache = {}
        def _get_kernel_col(idx):
            if idx in kernel_cache: # Cache hit
                kernel_col = kernel_cache.pop(idx)
                kernel_cache[idx] = kernel_col
                return kernel_col

            if len(kernel_cache) >= self.max_cache_size: # Truncate if exceed max size
                oldest_idx = next(iter(kernel_cache))
                del kernel_cache[oldest_idx]

            # Cache miss, compute the kernel and store it column-wise
            # Special case for gaussian kernel, we can use some kind of precomputation
            if is_gaussian:
                euclid_dist = np.maximum(X_square + X_square[idx] - 2.0 * (X @ X[idx]), 0.0)
                kernel_col = np.exp(-self._gamma * euclid_dist)
            else:
                kernel_col = self._compute_kernel(X, X[idx:idx+1]).ravel()

            kernel_cache[idx] = kernel_col
            return kernel_col


        # Implementation of procedure takeStep(i1, i2) from SMO pseudocode
        # We use the outer intercept b variable, the E array (SVM output minus some y value), and the w value for linear kernel
        non_zero_and_non_C_tracker = np.zeros(n_samples, dtype=bool)
        non_zero_and_non_C = np.empty(0, dtype=np.intp)
        found_new_value_tracked = False
        K_i2 = None

        def _take_step(i1, i2):
            nonlocal b, E, w, K_i2, found_new_value_tracked
            if i1 == i2:
                return False

            alph1 = alphas[i1]
            alph2 = alphas[i2]
            y1 = y[i1]
            y2 = y[i2]
            E1 = E[i1]
            E2 = E[i2]
            s = y1 * y2

            # Compute L, H via equations (13) and (14)
            if y1 != y2:
                L = max(0.0, alph2 - alph1)
                H = min(self.C, self.C + alph2 - alph1)
            else:
                L = max(0.0, alph1 + alph2 - self.C)
                H = min(self.C, alph1 + alph2)

            if L == H:
                return False

            if K_i2 is None:
                K_i2 = _get_kernel_col(i2)
            k11 = K_diag[i1]
            k12 = K_i2[i1]
            k22 = K_diag[i2]
            eta = k11 + k22 - 2.0 * k12

            if eta > 0:
                a2 = alph2 + y2 * (E1 - E2) / eta
                if a2 < L:
                    a2 = L
                elif a2 > H:
                    a2 = H
            else: 
                # For this one we follow the standford pseudocode if not the standard kernel, because standard one is too slow
                if self.kernel not in ('linear', 'poly', 'rbf', 'gaussian'):
                    f1 = y1 * (E1 + b) - alph1 * k11 - s * alph2 * k12
                    f2 = y2 * (E2 + b) - s * alph1 * k12 - alph2 * k22
                    L1 = alph1 + s * (alph2 - L)
                    H1 = alph1 + s * (alph2 - H)
                    Lobj = L1 * f1 + L * f2 + 0.5 * (L1 ** 2) * k11 + 0.5 * (L ** 2) * k22 + s * L * L1 * k12
                    Hobj = H1 * f1 + H * f2 + 0.5 * (H1 ** 2) * k11 + 0.5 * (H ** 2) * k22 + s * H * H1 * k12

                    if Lobj < Hobj - self._SMO_EPSILON:
                        a2 = L
                    elif Lobj > Hobj + self._SMO_EPSILON:
                        a2 = H
                    else:
                        a2 = alph2

                else:
                    return False


            if abs(a2 - alph2) < self._SMO_EPSILON: # For this one we follow the standford pseudocode, because standard one is too slow
                return False

            a1 = alph1 + s * (alph2 - a2)

            # Update threshold to reflect change in Lagrange multipliers
            b1 = b - E1 - y1 * (a1 - alph1) * k11 - y2 * (a2 - alph2) * k12
            b2 = b - E2 - y1 * (a1 - alph1) * k12 - y2 * (a2 - alph2) * k22
            
            b_old = b
            if 0 < a1 < self.C:
                b = b1
            elif 0 < a2 < self.C:
                b = b2
            else:
                b = 0.5 * (b1 + b2)

            # Update weight vector to reflect change in a1 & a2, if SVM is linear
            if self.kernel == 'linear':
                w += y1 * (a1 - alph1) * X[i1] + y2 * (a2 - alph2) * X[i2]

            # Update error cache using new Lagrange multipliers
            K_i1 = _get_kernel_col(i1)
            E += y1 * (a1 - alph1) * K_i1 + y2 * (a2 - alph2) * K_i2 + (b - b_old)

            # Store a1 and a2 in the alpha array
            alphas[i1] = a1
            alphas[i2] = a2

            nb1, nb2 = (0 < a1 < self.C), (0 < a2 < self.C)
            if non_zero_and_non_C_tracker[i1] != nb1 or non_zero_and_non_C_tracker[i2] != nb2:
                non_zero_and_non_C_tracker[i1] = nb1
                non_zero_and_non_C_tracker[i2] = nb2
                found_new_value_tracked = True

            return True


        # Implementation of main routine from pseudocode
        w = None
        if self.kernel == 'linear':
            w = np.zeros(n_features)

        num_changed = 0
        examine_all = True
        iteration = 0
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            if examine_all:
                indices_to_check = range(n_samples) # loop I over all training examples
            else:
                indices_to_check = np.flatnonzero(non_zero_and_non_C_tracker) # loop I over examples where alpha is not 0 & not C
            
            # This is the implementation of examineExample(i2), done for each indices 'i'
            for i2 in indices_to_check:
                y2 = y[i2]
                E2 = E[i2]
                alph2 = alphas[i2]
                r2 = E2 * y2
                if (r2 < -self.tolerance and alph2 < self.C) or (r2 > self.tolerance and alph2 > 0):
                    
                    K_i2 = None
                    step_taken = False

                    if found_new_value_tracked:
                        non_zero_and_non_C = np.flatnonzero(non_zero_and_non_C_tracker)
                        found_new_value_tracked = False

                    # number of non-zero & non-C alpha > 1
                    if len(non_zero_and_non_C) > 1:
                        
                        # result of second choice heuristic (section 2.2)
                        # We make the |E1 - E2| as maximum as possible for this heuristic
                        if E2 > 0:
                            i1 = non_zero_and_non_C[np.argmin(E[non_zero_and_non_C])]
                        else:
                            i1 = non_zero_and_non_C[np.argmax(E[non_zero_and_non_C])]

                        step_taken = _take_step(i1, i2)
                        
                    # loop over all non-zero and non-C alpha, starting at a random point
                    if not step_taken and len(non_zero_and_non_C) > 0:
                        start = int(self._rng.integers(len(non_zero_and_non_C)))
                        for offset in range(len(non_zero_and_non_C)):
                            i1 = non_zero_and_non_C[(start + offset) % len(non_zero_and_non_C)]
                            if _take_step(i1, i2):
                                step_taken = True
                                break
                                
                    # loop over all possible i1, starting at a random point
                    if not step_taken:
                        start = int(self._rng.integers(n_samples))
                        for offset in range(n_samples):
                            i1 = (start + offset) % n_samples
                            if _take_step(i1, i2):
                                step_taken = True
                                break
                    
                    # Equivalent with examineExample returning 1 in pseudocode
                    if step_taken:
                        num_changed += 1

            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
                
            iteration += 1

        # Main routin of SMO is done, find all alphas which is not zero and make it the support vectors
        sv_indices = alphas > self._SMO_EPSILON
        self._support = np.where(sv_indices)[0]
        self._support_vectors = X[sv_indices]
        self._dual_coef = (alphas[sv_indices] * y[sv_indices])
        self._intercept = b
        self._coef = w # For linear kernel only

        return self

    # Evaluate decision function for the samples in X
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)

        # For linear kernel we can use the weight vector immediately
        if self.kernel == 'linear' and self._coef is not None:
            return X @ self._coef + self._intercept
        
        # If we found no support vector just return the intercept vector
        if len(self._support_vectors) == 0:
            return np.full(X.shape[0], self._intercept)

        # For non-linear kernel, we compute the kernel and use dual coefficient + intercept formula to get the decision
        # Implementation is batched in case it is too big and memory not enough
        results = np.empty(X.shape[0], dtype=float)
        for start in range(0, X.shape[0], self._DECISION_BATCH_SIZE):
            end = start + self._DECISION_BATCH_SIZE
            K_batch = self._compute_kernel(X[start:end], self._support_vectors)
            results[start:end] = np.dot(K_batch, self._dual_coef) + self._intercept
            
        return results

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


# Plot the decision boundary of 2D data based on a fitted SVC model
def plot_decision_boundary(model, X, y, label1="Feature 1", label2="Feature 2", title="SVM Decision Boundary Visualisation"):
    if not isinstance(model, SVC):
        raise TypeError(f"Unknown model parameter passed, the model should be an SVC model.")

    X = np.asarray(X, dtype=float)
    if X.shape[1] != 2:
        raise NotImplementedError(f"Only supports 2D data on X, the X dimension is {X.shape[1]}.")
    
    y = np.asarray(y).reshape(-1)
    if y.shape[0] != X.shape[0]:
        raise ValueError(f"Number of samples mismatch, X has {X.shape[0]} samples but y has {y.shape[0]} samples.")

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter X points using different color for each class
    y_colors = np.where(y == model._classes[0], -1, 1)
    ax.scatter(X[:, 0], X[:, 1], c=y_colors, cmap=plt.cm.coolwarm, s=30, edgecolors='k')
    
    # Show the actual support vector with different style
    if model._support_vectors is not None and len(model._support_vectors) > 0:
        ax.scatter(model._support_vectors[:, 0], model._support_vectors[:, 1], s=150, facecolors='none', edgecolors='k', linewidths=1.5, label='Support Vectors')

    # Find hyperplane by evaluating the decision function, based on the X-Y coordinate
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.decision_function(grid_points)
    Z = Z.reshape(xx.shape)

    # Draw the hyperplane and the line intersecting with each outer data points
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], alpha=0.5, linestyles=['--', '-', '--'])

    ax.set_title(title)
    ax.set_xlabel(label1)
    ax.set_ylabel(label2)
    ax.legend()
    plt.show()