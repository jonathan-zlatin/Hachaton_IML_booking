from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np


def preprocess(X: pd.DataFrame, y: pd.DataFrame):
    X.drop("checkin_date", axis=1, inplace=True)
    X.drop("hotel_id", axis=1, inplace=True)
    X.drop("hotel_country_code", axis=1, inplace=True)
    X.drop("original_payment_currency", axis=1, inplace=True)
    X.drop("language", axis=1, inplace=True)
    pd.get_dummies(X, columns=["charge_option"])
    return X, y[X.index]


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, y_train = preprocess(X, y)
    print()

