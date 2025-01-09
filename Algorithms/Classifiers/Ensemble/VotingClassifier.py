import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

class VotingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, estimators, voting='hard', weights=None):
        self.estimators = estimators
        self.voting = voting
        self.weights = weights

    def fit(self, X, y):
        # Validate input
        X, y = check_X_y(X, y)

        # Store input data
        self.classes_ = np.unique(y)
        self.estimators_ = []

        # Train each estimator
        for _, estimator in self.estimators:
            cloned_estimator = clone(estimator)
            cloned_estimator.fit(X, y)
            self.estimators_.append(cloned_estimator)

        return self

    def predict(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['estimators_'])

        # Validate input
        X = check_array(X)

        if self.voting == 'hard':
            # Collect predictions from each estimator
            predictions = np.asarray([estimator.predict(X) for estimator in self.estimators_]).T

            # Perform weighted majority voting
            if self.weights:
                weighted_votes = np.apply_along_axis(lambda x: np.bincount(x, weights=self.weights, minlength=len(self.classes_)), axis=1, arr=predictions)
                return self.classes_[np.argmax(weighted_votes, axis=1)]
            else:
                majority_votes = np.apply_along_axis(lambda x: np.bincount(x, minlength=len(self.classes_)).argmax(), axis=1, arr=predictions)
                return majority_votes

        elif self.voting == 'soft':
            # Collect probabilities from each estimator
            probas = np.asarray([estimator.predict_proba(X) for estimator in self.estimators_])

            # Perform weighted averaging of probabilities
            if self.weights:
                avg_proba = np.average(probas, axis=0, weights=self.weights)
            else:
                avg_proba = np.mean(probas, axis=0)

            return self.classes_[np.argmax(avg_proba, axis=1)]

    def predict_proba(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['estimators_'])

        if self.voting != 'soft':
            raise AttributeError("predict_proba is not available when voting='hard'")

        # Validate input
        X = check_array(X)

        # Collect probabilities from each estimator
        probas = np.asarray([estimator.predict_proba(X) for estimator in self.estimators_])

        # Perform weighted averaging of probabilities
        if self.weights:
            avg_proba = np.average(probas, axis=0, weights=self.weights)
        else:
            avg_proba = np.mean(probas, axis=0)

        return avg_proba