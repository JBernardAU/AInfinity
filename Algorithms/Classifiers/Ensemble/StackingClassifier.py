import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.model_selection import KFold

class StackingClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimators=None, meta_estimator=None, cv=5, random_state=None):
        self.base_estimators = base_estimators
        self.meta_estimator = meta_estimator
        self.cv = cv
        self.random_state = random_state

    def fit(self, X, y):
        # Validate input
        X, y = check_X_y(X, y)

        # Initialize variables
        self.base_estimators_ = [clone(est) for est in self.base_estimators]
        self.meta_estimator_ = clone(self.meta_estimator)
        self.kf_ = KFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)

        # Create a matrix to hold out-of-fold predictions
        oof_predictions = np.zeros((X.shape[0], len(self.base_estimators_)))

        # Train base estimators using cross-validation
        for i, estimator in enumerate(self.base_estimators_):
            for train_idx, val_idx in self.kf_.split(X, y):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train = y[train_idx]

                # Fit the estimator and predict on validation set
                estimator.fit(X_train, y_train)
                oof_predictions[val_idx, i] = estimator.predict(X_val)

        # Fit the meta-estimator on the out-of-fold predictions
        self.meta_estimator_.fit(oof_predictions, y)

        # Refit all base estimators on the full dataset
        for estimator in self.base_estimators_:
            estimator.fit(X, y)

        return self

    def predict(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['base_estimators_', 'meta_estimator_'])

        # Validate input
        X = check_array(X)

        # Gather predictions from all base estimators
        base_predictions = np.column_stack([estimator.predict(X) for estimator in self.base_estimators_])

        # Use the meta-estimator to make the final prediction
        return self.meta_estimator_.predict(base_predictions)

    def predict_proba(self, X):
        # Check if the classifier is fitted
        check_is_fitted(self, ['base_estimators_', 'meta_estimator_'])

        # Validate input
        X = check_array(X)

        # Gather predictions from all base estimators
        base_predictions = np.column_stack([estimator.predict(X) for estimator in self.base_estimators_])

        # Use the meta-estimator to predict probabilities
        return self.meta_estimator_.predict_proba(base_predictions)
