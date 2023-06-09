from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor


class BaselineModel:
    def __init__(self, estimator, n_estimators):
        self.clf = AdaBoostClassifier(base_estimator=estimator, n_estimators=n_estimators)
        # self.clf = AdaBoostRegressor(base_estimator=estimator, n_estimators=n_estimators)
        #self.clf = estimator

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def loss(self, X, y):
        return self.clf.score(X, y)
