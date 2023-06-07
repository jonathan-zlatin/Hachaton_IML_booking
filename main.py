from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    print()

