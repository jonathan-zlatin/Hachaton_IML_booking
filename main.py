from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from ModelBaseline import BaselineModel
import plotly.graph_objects as go
import plotly.io as pio
from sklearn.tree import DecisionTreeClassifier
from enum import Enum
pio.renderers.default = 'browser'
class ModelType(Enum):
    CLASSIFIER = 0,
    REGRESSION = 1


def preprocess(X: pd.DataFrame, y: pd.DataFrame):
    # Has correlation with nationality
    X.drop(["h_customer_id", "customer_nationality", "origin_country_code", "h_booking_id",
            "language", "original_payment_method", "original_payment_currency", "is_user_logged_in",
            "cancellation_policy_code", "hotel_area_code", "booking_datetime", "checkin_date", "checkout_date",
            "hotel_live_date", "request_nonesmoke", 'request_latecheckin', "request_highfloor", "request_twinbeds",
            "request_airport","request_earlycheckin", "request_largebed","hotel_country_code", "charge_option",
            "guest_nationality_country_name", "original_payment_type",
            "hotel_brand_code", "hotel_chain_code", "hotel_city_code"
            ], axis=1, inplace=True)
    X["accommadation_type_name"] = np.where(X["accommadation_type_name"].isin["Hotel",
    "Guest House / Bed & Breakfast", "Hostel", "Resort"], X["accommadation_type_name"], 'other')
    X["charge_option"] = np.where(X["accommadation_type_name"] == "Pay at Check-in", "Pay Later",
                                  X["accommadation_type_name"])
    X["original_payment_type"] = np.where(X["original_payment_type"].isin["Visa",
    "MasterCard", "American Express"], X["accommadation_type_name"], 'other')

    X = pd.get_dummies(X, columns=["accommadation_type_name", "charge_option", "original_payment_type"])
    return X, y[X.index]


if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    X, y = df.drop("cancellation_datetime", axis=1), df["cancellation_datetime"]
    y = np.where(y.isnull(), 0, 1)
    X, y = preprocess(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    # corr = X_train.corr()
    # sns.heatmap(corr, annot=True, cmap=plt.cm.Reds, linewidth=.5)
    # print(corr)
    # plt.show()

    error = []
    estimator = DecisionTreeClassifier(max_depth=1)
    DecisionTreeClassifier()
    for i in list([50, 100, 200, 300, 500]):
        model = BaselineModel(ModelType.CLASSIFIER, estimator, i)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        error.append(model.loss(X_test, y_test))

    go.Figure(data=[go.Scatter(x=list([50, 100, 200, 300, 500]), y=error, name="Train error", mode="lines")],
              layout=go.Layout(title="",
                               xaxis_title="estimator number",
                               yaxis_title="Misclassification error")).show()
