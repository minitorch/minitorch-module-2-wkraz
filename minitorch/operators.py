"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable, Optional

#
# Implementation of a prelude of elementary functions.


# Mathematical functions:
# - mul
def mul(x: float, y: float) -> float:
    return x * y


# - id
def id(x: float) -> float:
    return x


# - add
def add(x: float, y: float) -> float:
    return x + y


# - neg
def neg(x: float) -> float:
    return -x


# - lt
def lt(x: float, y: float) -> bool:
    return x < y


# - eq
def eq(x: float, y: float) -> bool:
    return x == y


# - max
def max(x: float, y: float) -> float:
    if x >= y:
        return x
    return y


# - is_close
def is_close(x: float, y: float, tol: float = 1e-2) -> bool:
    return abs(x - y) < tol


# - sigmoid
def sigmoid(x: float) -> float:
    if x >= 0:
        return (1.0) / (1.0 + math.exp(-x))
    else:
        return (math.exp(x)) / (1.0 + math.exp(x))


# - relu
def relu(x: float) -> float:
    return x if x > 0 else 0


# - log
def log(x: float) -> float:
    return math.log(x)


# - exp
def exp(x: float) -> float:
    return math.exp(x)


# - log_back
def log_back(x: float, d: float) -> float:
    return d / x


# - inv
def inv(x: float) -> float:
    return 1 / x


# - inv_back
def inv_back(x: float, d: float) -> float:
    return -d / (x**2)


# - relu_back
def relu_back(x: float, d: float) -> float:
    return d if x > 0 else 0


#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# ## Task 0.3

# Small practice library of elementary higher-order functions.


# Implement the following core functions
# - map
def map(fn: Callable[[float], float], lst: Iterable[float]) -> Iterable[float]:
    """Mapping an iterable to another iterable where each element is sent into a function

    Args:
    ----
        fn: A function takes in a float as an argument and returns a float
        lst: An iterable of floats

    Returns:
    -------
        An iterable of floats

    """
    return [fn(x) for x in lst]


# - zipWith
def zipWith(
    fn: Callable[[float, float], float], lst1: Iterable[float], lst2: Iterable[float]
) -> Iterable[float]:
    """Combines elements from 2 iterables given a function

    Args:
    ----
        fn: A function takes in 2 floats as an argument and returns a float
        lst1: An iterable of floats
        lst2: The second iterable of floats

    Returns:
    -------
        An iterable of floats

    """
    return [fn(x, y) for x, y in zip(lst1, lst2)]


# - reduce
def reduce(
    fn: Callable[[float, float], float],
    lst: Iterable[float],
    initial: Optional[float] = None,
) -> float:
    """Reduces an iterable to a single value given a function.

    Args:
    ----
        fn: A function that takes in 2 floats as an argument and returns a float.
        lst: An iterable of floats.
        initial: The initial value to use when the list is empty (default is None).

    Returns:
    -------
        A float.

    Raises:
    ------
        ValueError: If the list is empty and no initial value is provided.

    """
    iterator = iter(lst)
    if initial is not None:
        value = initial
    else:
        try:
            value = next(iterator)  # Start with the first element
        except StopIteration:
            raise ValueError("reduce() of an empty list with no initial value")

    for element in iterator:
        value = fn(value, element)

    return value


#
# Use these to implement
# - negList : negate a list
def negList(lst: Iterable[float]) -> Iterable[float]:
    """Negates all elements in an iterable using a map and the neg function

    Args:
    ----
        lst: An iterable of floats

    Returns:
    -------
        An iterable of floats where each element is negated

    """
    return map(neg, lst)


# - addLists : add two lists together
def addLists(lst1: Iterable[float], lst2: Iterable[float]) -> Iterable[float]:
    """Adds two lists together

    Args:
    ----
        lst1: An iterable of floats
        lst2: An iterable of floats

    Returns:
    -------
        An iterable of floats that is the lists added together

    """
    return zipWith(add, lst1, lst2)


# - sum: sum lists
def sum(lst: Iterable[float]) -> float:
    """Gets the sum of all elements in a list

    Args:
    ----
        lst: An iterable of floats

    Returns:
    -------
        A float that is the sum of all elements in the list

    """
    return reduce(add, lst, initial=0)


# - prod: take the product of lists
def prod(lst: Iterable[float]) -> float:
    """Gets the product of all elements in a list

    Args:
    ----
        lst: An iterable of floats

    Returns:
    -------
        A float that is the product of all elements in the list

    """
    return reduce(mul, lst)
