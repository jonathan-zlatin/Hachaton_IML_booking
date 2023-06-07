from sklearn import tree


class BaselineModel:
    def __init__(self):
        self.clf = tree.DecisionTreeClassifier()
        self.max_depth = 5

    def fit(self, X, y):
        self.clf = self.clf.fit(X, y)

    def predict(self, y):
        return self.clf.predict(y)

    def loss(self,X, y):
        return self.clf.score(X, y)