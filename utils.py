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

def validate_format(policy_format: str) -> bool:
    # regex = re.compile(r"")
    for ch in policy_format:
        if ch not in {"D", "P", "N", "_"} or not ch.isdigit():
            return False
    return True


def parse_cancellation_policy_old(policy: str) -> float:
    if not validate_format(policy):
        return 0

    *cancelled_ahead_policy, last_part_of_policy = policy.split("_")
    # Validate if no show
    if "D" not in last_part_of_policy:
        no_show_policy = last_part_of_policy
    for policy_part in cancelled_ahead_policy:
        days, after = policy_part.split("D")


    cancelled_before_x = cancelled_ahead_policy[0]
    if len(cancelled_ahead_policy) == 2:
        cancelled_after_x = cancelled_before_x[1]


def parse_policy_part(policy: str) -> tuple:
    cancel_ahead_pattern = re.compile(r"([0-9]{1,3})(D)([0-9]{1,3})([N,P])")
    no_show_pattern = re.compile(r"([0-9]{1,3})([N,P])")
    if (first_match := cancel_ahead_pattern.match(policy)) is None:
        if (second_match := no_show_pattern.match(policy)) is None:
            return None, None, None
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


def convert_tuples_to_list(tuples: list[tuple]) -> list:
    temp = [0, 0, 0, 0, 0]
    # for i, parsed in enumerate(reversed(tuples)):
    #     if len(parsed) == 2:  # No show fine
    #         temp[0] = parsed[0]
    #     elif len(parsed) == 3:
    #         if i == 1:
    #             temp[1] = parsed[0]
    #             temp[2] = parsed[1]
    #         elif i == 2:
    #             temp[3] = parsed[0]
    #             temp[4] = parsed[1]
    # if temp[0] <= temp[2]:
    #     temp[0] = temp[2]
    i = 0
    for parsed in reversed(tuples):
        temp[i] = parsed[0]
        i += 1
        if len(parsed) == 3:
            temp[i] = parsed[1]
            i += 1
    return temp

def calculate_score(row: pd.Series) -> int:
    cancellation_policy = parse_cancellation_policy(row["cancellation_policy_code"])
    # cancellation_policy = np.array([
    #     convert_policy_part_to_percent(policy_part, row)
    #     for policy_part in cancellation_policy
    # ])
    cancellation_policy = list(filter(
        lambda policy_part: len(policy_part) == 3
    , cancellation_policy))
    return np.sum([
        d * p for d, p, _ in cancellation_policy
    ])

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


# def fill_empty_fines(df: pd.DataFrame) -> pd.DataFrame:
#     df["second_period"] = np.where(df["second_period"].isnull(), 0, df["second_period"])
#     df["second_fine"] = np.where(df["second_fine"].isnull(), 0, df["second_fine"])
#     df["first_period"] = np.where(df["first_period"].isnull(), 0, df["first_period"])
#     df["first_fine"] = np.where(df["first_fine"].isnull(), df["second_fine"], df["first_fine"])
#     df["no_show_fine"] = np.where(df["no_show_fine"].isnull(), df["first_fine"], df["no_show_fine"])
#     return df


def bla_format_N2P(nights, row: pd.Series):
    format_N2P(nights, row)

# def bla_convert_to_percent(parsed, row):
#     convert_policy_part_to_percent(parsed, row)


if __name__ == "__main__":
    res = parse_policy_part("1D1N")
    df = pd.read_csv('agoda_cancellation_train.csv')
    # bla_convert_to_percent((1, "N"), df.loc[445])
    row = df.iloc[7]
    parsed = parse_cancellation_policy(row["cancellation_policy_code"])
    parsed = parsed[0]
    result = format_N2P(parsed[1], row)
    print("{}P".format(result))
    score = calculate_score(row)
    print(score)
    # convert_cancellation_code_to_columns(df["cancellation_policy_code"])
    df = convert_cancellation_code_to_columns(df)
    # fill_empty_fines(df)
    convert_tuples_to_list([(365, 100, "P"), (100, "P")])
