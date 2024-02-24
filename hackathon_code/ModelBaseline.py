from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor


class BaselineModel:
    """
    Baseline model for classification or regression using AdaBoost.

    Attributes:
    - clf: AdaBoost classifier or regressor.
    """

    def __init__(self, estimator, n_estimators):
        """
        Initialize the BaselineModel.

        Args:
        - estimator: The base estimator to be used in AdaBoost.
        - n_estimators (int): The number of estimators to use in AdaBoost.
        """
        self.clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estimators)
        # self.clf = AdaBoostRegressor(base_estimator=estimator, n_estimators=n_estimators)
        #self.clf = estimator

    def fit(self, X, y):
        """
        Fit the model to the given training data.

        Args:
        - X (array-like or sparse matrix): The training input samples.
        - y (array-like): The target values.

        Returns:
        None
        """
        self.clf = self.clf.fit(X, y)

    def predict(self, X):
        """
        Predict class labels or regression targets for the input data.

        Args:
        - X (array-like or sparse matrix): The input samples.

        Returns:
        - array-like: Predicted class labels or regression targets.
        """
        return self.clf.predict(X)

    def loss(self, X, y):
        """
        Compute the loss of the model on the given data.

        Args:
        - X (array-like or sparse matrix): The input samples.
        - y (array-like): The true target values.

        Returns:
        - float: The loss of the model.
        """
        return self.clf.score(X, y)
