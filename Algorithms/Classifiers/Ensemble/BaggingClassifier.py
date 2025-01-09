import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import resample


class BaggingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, n_estimators=10, max_samples=1.0, random_state=None):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

    def fit(self, X, y):
        # Validate input
        X, y = check_X_y(X, y)

        # Store input shape
        self.n_samples_, self.n_features_ = X.shape

        # Initialize random state
        self.rng_ = np.random.RandomState(self.random_state)

        # Create and train estimators
        self.estimators_ = []
        self.estimators_samples_ = []

        for _ in range(self.n_estimators):
            # Generate bootstrap sample
            indices = self.rng_.choice(self.n_samples_,
                                       size=int(self.max_samples * self.n_samples_),
                                       replace=True)
            X_sample, y_sample = X[indices], y[indices]

            # Clone the base estimator and fit it
            estimator = clone(self.base_estimator)
            estimator.fit(X_sample, y_sample)

            # Save the estimator and its sample indices
            self.estimators_.append(estimator)
            self.estimators_samples_.append(indices)

        return self

    def predict(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['estimators_', 'estimators_samples_'])

        # Validate input
        X = check_array(X)

        # Aggregate predictions from all estimators
        predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_]).T

        # Majority voting
        majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=np.max(x) + 1).argmax(),
                                             axis=1,
                                             arr=predictions)
        return majority_votes

    def predict_proba(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['estimators_', 'estimators_samples_'])

        # Validate input
        X = check_array(X)

        # Aggregate probabilities from all estimators
        proba = np.mean([estimator.predict_proba(X) for estimator in self.estimators_], axis=0)
        return proba
