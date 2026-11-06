"""Exact symmetric binomial test for paired binary outcomes, without SciPy."""
from math import comb


def exact_paired_p(gained: int, lost: int) -> float:
    if any(type(n) is not int or n < 0 for n in [gained, lost]):
        raise ValueError('Discordant counts must be nonnegative integers')
    total = gained + lost
    if not total:
        return 1.0
    tail = sum(comb(total, k) for k in range(min(gained,lost)+1))
    return min(1.0, (2*tail)/(2**total))
