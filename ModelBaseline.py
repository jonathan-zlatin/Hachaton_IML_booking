from sklearn import tree


class BaselineModel:
    def __init__(self, max_depth):
        self.clf = tree.DecisionTreeClassifier(max_depth=max_depth)
        self.max_depth = max_depth

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)

    def predict(self, X):
        return self.clf.predict(X)

    def loss(self, X, y):
        return self.clf.score(X, y)
