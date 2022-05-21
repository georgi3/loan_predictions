import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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


def winsorize_pandas(array, limits):
    return array.clip(lower=array.quantile(limits[0], interpolation='lower'),
                      upper=array.quantile(1-limits[1], interpolation='higher'))


class ToDenseTransformer:
    def transform(self, X, y=None):
        return  X.todense()

    def fit(self, X, y=None):
        return self


class ClfSwitcher(BaseEstimator):
    def __init__(self, estimator=LogisticRegression()):
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


# PLOTTING
def conf_matrix(confusion_matrix, title='Confusion Matrix'):
    conf_df = pd.DataFrame(confusion_matrix, range(2), range(2))
    plt.figure(figsize=(6,6))
    sns.set(font_scale=1.5)
    sns.heatmap(data=conf_df, annot=True, annot_kws={"size": 16}, fmt='g',  cmap='Greens', cbar=False)
    plt.title(title)




