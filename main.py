from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
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

    # Calculate the difference in days between 'checkin_date' and 'booking_datetime'
    X['days_before_checkin'] = (X['checkin_date'] - X['booking_datetime']).dt.days + 1

    X.drop(["h_customer_id", "customer_nationality", "origin_country_code",
            "language", "original_payment_method", "original_payment_currency", "is_user_logged_in",
            "cancellation_policy_code", "hotel_area_code", "booking_datetime", "checkin_date", "checkout_date",
            "hotel_live_date", "request_nonesmoke", 'request_latecheckin', "request_highfloor", "request_twinbeds",
            "request_airport", "request_earlycheckin", "request_largebed", "hotel_country_code", "charge_option",
            "guest_nationality_country_name",
            "hotel_brand_code", "hotel_chain_code", "hotel_city_code"
            ], axis=1, inplace=True)

    X["accommadation_type_name"] = np.where(X["accommadation_type_name"].isin(["Hotel",
                                                                               "Guest House / Bed & Breakfast",
                                                                               "Hostel", "Resort"]),
                                            X["accommadation_type_name"], 'other')
    X["charge_option"] = np.where(X["accommadation_type_name"] == "Pay at Check-in", "Pay Later",
                                  X["accommadation_type_name"])
    X["original_payment_type"] = np.where(X["original_payment_type"].isin(["Visa",
                                                                           "MasterCard", "American Express"]),
                                          X["accommadation_type_name"], 'other')

    X = pd.get_dummies(X, columns=["accommadation_type_name", "charge_option", "original_payment_type"])
    return X, y[X.index]


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    y = np.where(y.isnull(), 0, 1)
    X, y = preprocess(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    error = []
    estimator = tree.DecisionTreeClassifier(max_depth=30)
    for i in list([50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]):
        model = BaselineModel(estimator, i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error.append(model.loss(X_test, y_test))

    go.Figure(
        data=[go.Scatter(
            x=list([50, 100, 200, 300, 500, 1000, 2000, 3000, 4000, 5000]),
            y=error, name="Train error", mode="lines")],
        layout=go.Layout(title="",
                         xaxis_title="estimator number",
                         yaxis_title="Misclassification error")).show()

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cancellation_predictions = pd.DataFrame({"id": X_test["h_booking_id"], "cancellation": y_pred})

    # Save the DataFrame to a CSV file
    cancellation_predictions.to_csv("agoda_cancellation_prediction.csv", index=False)

    # Estimate loss of cancellation

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cost_of_cancellation_predictions = pd.DataFrame({"id": X_test["h_booking_id"], "predicted_selling_amount": y_pred})

    # Save the DataFrame to a CSV file
    cost_of_cancellation_predictions.to_csv("agoda_cost_of_cancellation.csv", index=False)

