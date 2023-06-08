import re


def validate_format(policy_format: str) -> bool:
    # regex = re.compile(r"")
    for ch in policy_format:
        if ch not in {"D", "P", "N", "_"} or not ch.isdigit():
            return False
    return True


def parse_cancellation_policy(policy: str) -> float:
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


def parse(policy: str) -> tuple:
    pattern = re.compile(r"([0-9]{1,3})(D)([0-9]{1,3})([N, P])")
    if (match := pattern.match(policy)) is None:
        return None, None, None
    return int(match.group(1)), int(match.group(3)), match.group(4)


if __name__ == "__main__":
    res = parse("1D1N")
    print()
