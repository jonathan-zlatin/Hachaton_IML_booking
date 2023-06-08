import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import utils
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from ModelBaseline import BaselineModel
import plotly.graph_objects as go
import plotly.io as pio
from sklearn import tree
import csv

pio.renderers.default = 'browser'


def preprocess(X: pd.DataFrame, y: pd.DataFrame):
    # Convert the 'checkin_date' and 'booking_datetime' columns to datetime type
    X['checkin_date'] = pd.to_datetime(X['checkin_date'])
    X['booking_datetime'] = pd.to_datetime(X['booking_datetime'])
    X = utils.convert_cancellation_code_to_columns(X)
    # Calculate the difference in days between 'checkin_date' and 'booking_datetime'
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days + 1

    X.drop(["h_customer_id", "customer_nationality", "origin_country_code",
            "language", "original_payment_currency", "is_user_logged_in",
            "hotel_area_code", "booking_datetime", "checkin_date", "checkout_date",
            "hotel_live_date", "request_nonesmoke", 'request_latecheckin', "request_highfloor", "request_twinbeds",
            "request_airport", "request_earlycheckin", "request_largebed", "hotel_country_code",
            "guest_nationality_country_name",
            "hotel_brand_code", "hotel_chain_code", "hotel_city_code"], axis=1, inplace=True)

    X["accommadation_type_name"] = np.where(X["accommadation_type_name"].isin(["Hotel",
                                                                               "Guest House / Bed & Breakfast",
                                                                               "Hostel", "Resort"]),
                                            X["accommadation_type_name"], 'other')
    X["charge_option"] = np.where(X["charge_option"] == "Pay at Check-in", "Pay Later",
                                  X["charge_option"])
    X["original_payment_method"] = np.where(X["original_payment_method"].isin(["Visa",
                                                                           "MasterCard", "American Express"]),
                                          X["original_payment_method"], 'other')

    X = pd.get_dummies(X, columns=["accommadation_type_name", "charge_option", "original_payment_type",
                                   "original_payment_method"])
    return X, y[X.index]

def replace_none_with_samples(data):
    # Iterate over each column
    for column in data.columns:
        # Get non-None values
        non_none_values = data[column].dropna()

        # Generate random samples from the non-None values
        samples = np.random.choice(non_none_values, size=data[column].isna().sum(), replace=True)

        # Replace None values with the generated samples
        data.loc[data[column].isna(), column] = samples

    return data


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    y = np.where(y.isnull(), 0, 1)
    X, y = preprocess(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    estimator = DecisionTreeClassifier(max_depth=1)
    model = BaselineModel(estimator, 200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_pred, y_test, average='macro')
    print("The f1 Score For ADA trees is:" + str(f1))

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cancellation_predictions = pd.DataFrame({"id": X_test["h_booking_id"], "cancellation": y_pred})

    # Save the DataFrame to a CSV file
    cancellation_predictions.to_csv("agoda_cancellation_prediction.csv", index=False)

    # Estimate loss of cancellation

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cost_of_cancellation_predictions = pd.DataFrame({"id": X_test["h_booking_id"], "predicted_selling_amount": y_pred})

    # Save the DataFrame to a CSV file
    cost_of_cancellation_predictions.to_csv("agoda_cost_of_cancellation.csv", index=False)

    # Q3
    pearsons = list()
    params = list()
    for param in X:
        pearson_corr = np.cov(X[param], y)[0, 1] / (np.std(X[param]) * np.std(y))
        pearsons.append(pearson_corr)
        params.append(param)
        print(f"The Pearson corr of {param} is : {pearson_corr}")

    max_ = np.argmax(pearsons)
    min_ = np.argmin(pearsons)
    print(f"And The Winner Is {params[max_]} with pearson corr of : {np.max(pearsons)}")
    print(f"And The Losser Is {params[min_]} with pearson corr of : {np.min(pearsons)}")