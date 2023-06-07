from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def preprocess(X: pd.DataFrame, y: pd.DataFrame):
    X.drop("hotel_id", axis=1, inplace=True)
    X.drop("hotel_country_code", axis=1, inplace=True)
    X["charge_option"] = X["charge_option"].replace("Pay Now", 0)
    X["charge_option"] = X["charge_option"].replace("Pay Later", 1)
    X["charge_option"] = X["charge_option"].replace("Pay at Check-in", 2)
    X.drop("h_customer_id", axis=1, inplace=True)
    X.drop("customer_nationality", axis=1, inplace=True)
    X.drop("origin_country_code", axis=1, inplace=True)
    # Has correlation with nationality
    X.drop("language", axis=1, inplace=True)
    X.drop("original_payment_method", axis=1, inplace=True)
    X.drop("original_payment_currency", axis=1, inplace=True)
    X.drop("is_user_logged_in", axis=1, inplace=True)
    X.drop("cancellation_policy_code", axis=1, inplace=True)
    X["is_first_booking"] = X["is_first_booking"].replace("TRUE", 1)
    X["is_first_booking"] = X["is_first_booking"].replace("FALSE", 0)
    X.drop("hotel_area_code", axis=1, inplace=True)
    X.drop("hotel_brand_code", axis=1, inplace=True)
    X.drop("hotel_chain_code", axis=1, inplace=True)
    X.drop("hotel_city_code", axis=1, inplace=True)
    # X = pd.get_dummies(X, columns=["hotel_id", "hotel_country_code", "accommadation_type_name", "charge_option",
    #                            "h_customer_id", "customer_nationality", "guest_nationality_country_name",
    #                            "origin_country_code", "language", "original_payment_method", "original_payment_type",
    #                            "original_payment_currency", "is_first_booking", "hotel_area_code",
    #                            "hotel_brand_code", "hotel_chain_code", "hotel_city_code"])
    # X["is_cancelled"] = X["cancellation_datetime"].apply(lambda x: 1 if x is not None else 0)
    return X, y[X.index]


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    y = y.apply(lambda x: 1 if x is not None else 0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train, y_train = preprocess(X, y)
    corr = X_train.corr()
    sns.heatmap(corr, annot=True, cmap=plt.cm.Reds, linewidth=.5)
    print(corr)
    plt.show()

