#Replace these library calls with classes
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

class GradientBoostingMachine:
    def __init__(self, library="xgboost", **params):
        """
        Initialize the Gradient Boosting Machine with the specified library.

        Args:
            library (str): The backend library to use ("xgboost", "lightgbm", "catboost").
            **params: Hyperparameters specific to the chosen library.
        """
        supported_libraries = ["xgboost", "lightgbm", "catboost"]
        if library not in supported_libraries:
            raise ValueError(f"Unsupported library: {library}. Choose from {supported_libraries}.")

        self.library = library
        self.params = params
        self.model = None

    def fit(self, X, y):
        """
        Train the model on the given data.

        Args:
            X (array-like or DataFrame): Features for training.
            y (array-like): Target values for training.
        """
        if self.library == "xgboost":
            dtrain = xgb.DMatrix(X, label=y)
            self.model = xgb.train(self.params, dtrain)

        elif self.library == "lightgbm":
            self.model = lgb.train(self.params, lgb.Dataset(X, label=y))

        elif self.library == "catboost":
            self.model = cb.CatBoost(self.params)
            self.model.fit(X, y, verbose=False)

    def predict(self, X):
        """
        Make predictions using the trained model.

        Args:
            X (array-like or DataFrame): Features for prediction.

        Returns:
            array-like: Predicted values.
        """
        if self.library == "xgboost":
            dtest = xgb.DMatrix(X)
            return self.model.predict(dtest)

        elif self.library == "lightgbm":
            return self.model.predict(X)

        elif self.library == "catboost":
            return self.model.predict(X)

    def save_model(self, path):
        """
        Save the trained model to a file.

        Args:
            path (str): File path to save the model.
        """
        if self.library == "xgboost":
            self.model.save_model(path)

        elif self.library == "lightgbm":
            self.model.save_model(path)

        elif self.library == "catboost":
            self.model.save_model(path)

    def load_model(self, path):
        """
        Load a model from a file.

        Args:
            path (str): File path to load the model from.
        """
        if self.library == "xgboost":
            self.model = xgb.Booster()
            self.model.load_model(path)

        elif self.library == "lightgbm":
            self.model = lgb.Booster(model_file=path)

        elif self.library == "catboost":
            self.model = cb.CatBoost()
            self.model.load_model(path)

# Example Usage:
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_regression

    # Generate some example data
    X, y = make_regression(n_samples=1000, n_features=20, noise=0.1)

    # Initialize and train a model using LightGBM
    gbm = GradientBoostingMachine(library="lightgbm", objective="regression")
    gbm.fit(X, y)

    # Predict on new data
    predictions = gbm.predict(X[:10])
    print("Predictions:", predictions)
