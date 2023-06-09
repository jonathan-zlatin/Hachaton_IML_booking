import pandas as pd
import re
import numpy as np


def format_N2P(nights: int, invitation: pd.Series):
    # Calculate the number of days the customer ordered
    # change to dt format
    invitation['checkin_date'] = pd.to_datetime(invitation['checkin_date'])
    invitation['checkout_date'] = pd.to_datetime(invitation['checkout_date'])

    days_stay = (invitation['checkout_date'] - invitation['checkin_date']).days

    price = invitation["original_selling_amount"]
    price_pr_night = price / days_stay

    # calculate the ratio price
    total_pay = price_pr_night * nights
    percent = int((total_pay / price) * 100)
    return percent


def parse_policy_part(policy: str) -> tuple:
    cancel_ahead_pattern = re.compile(r"([0-9]{1,3})(D)([0-9]{1,3})([N,P])")
    no_show_pattern = re.compile(r"([0-9]{1,3})([N,P])")
    first_match = cancel_ahead_pattern.match(policy)
    if first_match is None:
        second_match = no_show_pattern.match(policy)
        if second_match is None:
            return 0, "P"
        return int(second_match.group(1)), second_match.group(2)
    return int(first_match.group(1)), int(first_match.group(3)), first_match.group(4)


def parse_cancellation_policy(policy: str) -> list[tuple]:
    policy_parts = policy.split("_")
    policy_parts = [
        parse_policy_part(policy_part) for policy_part in policy_parts
    ]
    if len(policy_parts) == 1:
        if len(policy_parts[0]) == 3:  # No show
            policy_parts.append((policy_parts[0][1], policy_parts[0][2]))
    return policy_parts


def convert_policy_parts_to_percent(policy_parts: list[tuple], row: pd.Series) -> list:
    # Expects len of list >= 2
    temp = [0, 0, 0, 0, 0]
    i = 0
    for policy_part in reversed(policy_parts):
        unit = policy_part[-1]
        x = format_N2P(policy_part[-2], row) if unit == "N" else policy_part[-2]
        temp[i] = x
        i += 1
        if len(policy_part) == 3:
            temp[i] = policy_part[-3]
            i += 1
    return temp


def convert_cancellation_code_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["cancellation_policy_code"] = np.where(
        df["cancellation_policy_code"] == "UNKNOWN", "0P", df["cancellation_policy_code"]
    )
    df["cancellation_policy_code"] = df.apply(
        lambda row: convert_policy_parts_to_percent(parse_cancellation_policy(
            row["cancellation_policy_code"]
        ), row), axis=1
    )
    df = df.join(pd.DataFrame(df.pop("cancellation_policy_code").values.tolist()).rename(columns={
        0: "no_show_fine", 1: "first_fine", 2: "first_period", 3: "second_fine", 4: "second_period"
    }))
    return df


if __name__ == "__main__":
    res = parse_policy_part("1D1N")
    df = pd.read_csv('agoda_cancellation_train.csv')
    # bla_convert_to_percent((1, "N"), df.loc[445])
    row = df.iloc[7]
    parsed = parse_cancellation_policy(row["cancellation_policy_code"])
    parsed = parsed[0]
    result = format_N2P(parsed[1], row)
    print("{}P".format(result))
    df = convert_cancellation_code_to_columns(df)
