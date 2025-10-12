from itertools import product
import string


def generate_subsets(n) -> list[str]:
    """
    Generate all subsets of n circles as bitmasks.
    Each subset is an n-length string of 0/1.
    """
    return ["".join(bits) for bits in product("01", repeat=n)]


def label_subsets(n: int) -> dict[str, str]:
    labels = list(string.ascii_uppercase[:n])
    subsets = generate_subsets(n)
    result = {}
    for mask in subsets:
        members = [labels[i] for i, bit in enumerate(mask) if bit == "1"]
        result[mask] = "{" + ",".join(members) + "}" if members else "∅"
    return result
