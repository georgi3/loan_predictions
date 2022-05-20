import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB

# data engineering
def comb_income(data, cols):
    data['CombinedIncome'] = data[cols].sum(axis=1)
    data = data.drop(cols, axis=1)
    return np.log(data)


def to_drop(data, cols):
    return data[:, len(cols):]  # (cols, axis=1, inplace=True)


# data separation
def data_separator(data, cols):
    return data[cols]


# data transformation
def term_transformer(data, col):
    return (data[col] / 12).to_frame()


def log_transformer(data, cols):
    return np.log(data[cols])


class ToDenseTransformer:
    def transform(self, X, y=None):
        return  X.todense()

    def fit(self, X, y=None):
        return self


class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator=BernoulliNB()):
        self.estimator = estimator

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)




