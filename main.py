import pandas as pd
import numpy as np
from hackathon_code import task_1
from hackathon_code import task_2
import plotly.io as pio
import sys

pio.renderers.default = 'browser'

if __name__ == '__main__':
    np.random.seed(0)
    df = pd.read_csv('agoda_cancellation_train.csv')
    is_cancel = np.where(df["cancellation_datetime"].isnull(), 0, 1)
    dict = task_1.get_mean_dictionary(df)
    model1 = task_1.run_task_1(df, dict)

    # Q1
    Agoda_Test_1 = pd.read_csv(sys.argv[1])
    X_test1 = task_1.preprocess(Agoda_Test_1, None, dict)
    y_pred1 = model1.predict(X_test1)

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cancellation_predictions = pd.DataFrame({"id": X_test1["h_booking_id"], "cancellation": y_pred1})

    # Save the DataFrame to a CSV file
    cancellation_predictions.to_csv("agoda_cancellation_prediction.csv", index=False)

    # Q2
    dict2 = task_2.get_mean_dictionary(df)
    model2 = task_2.run_task_2(df, dict2)
    Agoda_Test_2 = pd.read_csv(sys.argv[2])
    X_test2 = task_2.preprocess(Agoda_Test_2.copy(deep=True), None, dict)
    y_pred2 = model2.predict(X_test2)
    Agoda_Test_2.insert(loc=df.columns.get_loc("original_selling_amount"), column="original_selling_amount",
                        value=y_pred2)

    X_ = task_1.preprocess(Agoda_Test_2, None, dict2)
    y_ = model1.predict(X_)
    for i in range(len(y_)):
        if y_[i] == 0:
            y_pred2[i] = -1

    # Create a DataFrame with the h_booking_id and its corresponding prediction
    cost_of_cancellation_predictions = pd.DataFrame(
        {"id": X_["h_booking_id"], "predicted_selling_amount": y_pred2})

    # Save the DataFrame to a CSV file
    cost_of_cancellation_predictions.to_csv("agoda_cost_of_cancellation.csv", index=False)

    # # Q3
    # pearsons = list()
    # params = list()
    # for param in df:
    #     pearson_corr = np.cov(df[param], is_cancel)[0, 1] / (np.std(df[param]) * np.std(is_cancel))
    #     pearsons.append(pearson_corr)
    #     params.append(param)
    #     print(f"The Pearson corr of {param} is : {pearson_corr}")
    #
    # max_ = np.argmax(pearsons)
    # min_ = np.argmin(pearsons)
    # print(f"And The Winner Is {params[max_]} with pearson corr of : {np.max(pearsons)}")
    # print(f"And The Losser Is {params[min_]} with pearson corr of : {np.min(pearsons)}")
