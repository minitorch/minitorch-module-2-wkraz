"""
Collection of the core mathematical operators used throughout the code base.
"""

import math
from typing import Callable, Iterable

# ## Task 0.1
#
# Implementation of a prelude of elementary functions.


def is_close(num1: float, num2: float) -> bool:
    return abs(num1 - num2) < 1e-2 


def mul(num1: float, num2: float) -> float:
    return num1 * num2


def max(num1: float, num2: float) -> float:
    return num1 if num1 > num2 else num2


def add(num1: float, num2: float) -> float:
    return num1 + num2


def neg(num1: float) -> float:
    return - num1


def id(num: float) -> float:
    return num


def inv(num: float) -> float:
    return 1.0 / num


def relu(num: float) -> float:
    return num if num > 0 else 0.0


def relu_back(num1: float, num2: float) -> float:
    return num2 if num1 > 0 else 0.0


def lt(num1: float, num2: float) -> float:
    return float(num1 < num2)


def eq(num1: float, num2: float) -> float:
    return float(num1 == num2)


def sigmoid(num: float) -> float:
    return 1.0/(1.0 + math.exp(-num)) if num >=0 else math.exp(num)/(1.0 + math.exp(num))


def log(num: float) -> float:
    return math.log(num + 1e-6)


def exp(num: float) -> float:
    return math.exp(num)


def log_back(num: float,  d: float) -> float:
    return d / num


def inv_back(num: float, d: float) -> float:
    return float(-d * 1 / (num * num))


# ## Task 0.3

# Small practice library of elementary higher-order functions.

# Implement the following core functions
# - map
# - zipWith
# - reduce
#
# Use these to implement
# - negList : negate a list
# - addLists : add two lists together
# - sum: sum lists
# - prod: take the product of lists


def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    return lambda x: [fn(el) for el in x]


def negList(ls: Iterable[float]) -> Iterable[float]:
    return map(neg)(ls)


def zipWith(
    fn: Callable[[float, float], float]
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    return lambda x, y: [fn(x[i], y[i]) for i in range(len(x))]


def addLists(ls1: Iterable[float], ls2: Iterable[float]) -> Iterable[float]:
    return zipWith(add)(ls1, ls2)


def reduce(
    fn: Callable[[float, float], float], start: float
) -> Callable[[Iterable[float]], float]:
    def foo(ls: Iterable[float]) -> float:
        old_result = start
        for el in ls:
            old_result = fn(el, old_result)
        
        return old_result

    return foo


def sum(ls: Iterable[float]) -> float:
    return reduce(add, 0.0)(ls)


def prod(ls: Iterable[float]) -> float:
    return reduce(mul, 1.0)(ls)