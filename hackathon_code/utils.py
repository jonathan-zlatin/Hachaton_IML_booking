import pandas as pd
import re
import numpy as np


def format_N2P(nights: int, invitation: pd.Series):
    """
        Calculate the percentage of the original booking price based on the number of nights stayed.

        Parameters:
            nights (int): The number of nights stayed.
            invitation (pd.Series): A pandas Series containing booking information,
            including the original selling amount.

        Returns:
            int: The calculated percentage of the original booking price.
        """
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
    """
        Parse a single cancellation policy part.

        Parameters:
            policy (str): A string representing a single cancellation policy part (e.g., "1D1N").

        Returns:
            tuple: A tuple containing the cancellation period and the time unit.
                For example, for the input "1D1N", the tuple returned would be (1, 'N').
        """
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
    """
      Parse a full cancellation policy string.

      Parameters:
          policy (str): A string representing the full cancellation policy (e.g., "1D1N_100P").

      Returns:
          list[tuple]: A list of tuples, where each tuple represents a cancellation policy part.
              For example, for the input "1D1N_100P", the returned list would be [(1, 'D'), (1, 'N'), (100, 'P')].
      """
    policy_parts = policy.split("_")
    policy_parts = [
        parse_policy_part(policy_part) for policy_part in policy_parts
    ]
    if len(policy_parts) == 1 and len(policy_parts[0]) == 3: # No show
        policy_parts.append((policy_parts[0][1], policy_parts[0][2]))
    return policy_parts


def convert_policy_parts_to_percent(policy_parts: list[tuple], _row: pd.Series) -> list:
    """
        Convert cancellation policy parts to percentages based on the actual booking data.

        Parameters:
            policy_parts (list[tuple]): A list of tuples representing cancellation policy parts.
            _row (pd.Series): A pandas Series containing booking information.

        Returns:
            list: A list containing the converted percentages.
        """
    # Expects len of list >= 2
    temp = [0, 0, 0, 0, 0]
    i = 0
    for policy_part in reversed(policy_parts):
        unit = policy_part[-1]
        x = format_N2P(policy_part[-2], _row) if unit == "N" else policy_part[-2]
        temp[i] = x
        i += 1
        if len(policy_part) == 3:
            temp[i] = policy_part[-3]
            i += 1
    return temp


def convert_cancellation_code_to_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
        Convert the cancellation policy code column in the DataFrame to separate columns representing different cancellation policy parameters.

        Parameters:
            df (pd.DataFrame): The DataFrame containing booking data, including the cancellation policy code column.

        Returns:
            pd.DataFrame: The modified DataFrame with separate columns for each cancellation policy parameter.
        """

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
    row = df.iloc[7]
    parsed = parse_cancellation_policy(row["cancellation_policy_code"])
    parsed = parsed[0]
    result = format_N2P(parsed[1], row)
    print("{}P".format(result))
    df = convert_cancellation_code_to_columns(df)
