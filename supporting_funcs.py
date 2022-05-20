import numpy as np


# data engineering
def comb_income(data, cols):
    data['CombinedIncome'] = data[cols].sum(axis=1)
    data = data.drop(cols, axis=1)
    return np.log(data)


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
