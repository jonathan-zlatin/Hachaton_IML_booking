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
    return [
        parse_policy_part(policy_part) for policy_part in policy_parts
    ]


def convert_policy_part_to_percent(policy_part: tuple, row: pd.Series):
    unit = policy_part[-1]
    if unit == "N":
        if len(policy_part) == 2:
            return format_N2P(policy_part[0], row), "P"
        return policy_part[0], format_N2P(policy_part[1], row), "P"
    return policy_part


def calculate_score(row: pd.Series) -> int:
    cancellation_policy = parse_cancellation_policy(row["cancellation_policy_code"])
    cancellation_policy = np.array([
        convert_policy_part_to_percent(policy_part, row)
        for policy_part in cancellation_policy
    ])
    cancellation_policy = list(filter(
        lambda policy_part: len(policy_part) == 3
    , cancellation_policy))
    return np.sum([
        d * p for d, p, _ in cancellation_policy
    ])


if __name__ == "__main__":
    res = parse_policy_part("1D1N")
    df = pd.read_csv('agoda_cancellation_train.csv')
    row = df.iloc[7]
    parsed = parse_cancellation_policy(row["cancellation_policy_code"])
    parsed = parsed[0]
    result = format_N2P(parsed[1], row)
    print("{}P".format(result))
    score = calculate_score(row)
    print(score)
