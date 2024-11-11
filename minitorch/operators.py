"""Collection of the core mathematical operators used throughout the code base."""

import math

# ## Task 0.1
from typing import Callable, Iterable

#
# Implementation of a prelude of elementary functions.

# Mathematical functions:
# - mul
# - id
# - add
# - neg
# - lt
# - eq
# - max
# - is_close
# - sigmoid
# - relu
# - log
# - exp
# - log_back
# - inv
# - inv_back
# - relu_back
#
# For sigmoid calculate as:
# $f(x) =  \frac{1.0}{(1.0 + e^{-x})}$ if x >=0 else $\frac{e^x}{(1.0 + e^{x})}$
# For is_close:
# $f(x) = |x - y| < 1e-2$


# TODO: Implement for Task 0.1.
def mul(x: float, y: float) -> float:
    """Multiply two numbers, x and y."""
    return x * y


def id(x: float) -> float:
    """Identity function, returns x itself."""
    return x


def add(x: float, y: float) -> float:
    """Add two numbers, x and y."""
    return x + y


def neg(x: float) -> float:
    """Negate a number, x."""
    return -1.0 * x


def lt(x: float, y: float) -> float:
    """Check if x is less than y. Return 1.0 if true, 0.0 otherwise."""
    return 1.0 if x < y else 0.0


def eq(x: float, y: float) -> float:
    """Check if x is equal to y. Return 1.0 if true, 0.0 otherwise."""
    return 1.0 if x == y else 0.0


def max(x: float, y: float) -> float:
    """Return the maximum of x and y."""
    return x if x > y else y


def is_close(x: float, y: float) -> bool:
    """Check if x is close to y."""
    return abs(x - y) < 1e-2


def sigmoid(x: float) -> float:
    """Return the sigmoid of x."""
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


def relu(x: float) -> float:
    """Return the ReLU of x."""
    return max(0.0, x)


def log(x: float) -> float:
    """Return the natural logarithm of x."""
    return math.log(x)


def exp(x: float) -> float:
    """Return the exponential of x."""
    return math.exp(x)


def log_back(x: float, d: float) -> float:
    """Return the derivative of the natural logarithm of x."""
    return d / x


def inv(x: float) -> float:
    """Return the inverse of x."""
    return 1 / x


def inv_back(x: float, d: float) -> float:
    """Return the derivative of the inverse of x."""
    return -d / (x**2)


def relu_back(x: float, d: float) -> float:
    """Return the derivative of the ReLU of x."""
    return d if x > 0 else 0.0


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


# TODO: Implement for Task 0.3.
def map(fn: Callable[[float], float]) -> Callable[[Iterable[float]], Iterable[float]]:
    """Implement a higher-order map function."""

    def map_fc(array: Iterable[float]) -> Iterable[float]:
        return [fn(x) for x in array]

    return map_fc


def zipWith(
    fn: Callable[[float, float], float],
) -> Callable[[Iterable[float], Iterable[float]], Iterable[float]]:
    """Implement a higher-order zipWith function."""

    def zipWith_fc(array1: Iterable[float], array2: Iterable[float]) -> Iterable[float]:
        return [fn(x, y) for x, y in zip(array1, array2)]

    return zipWith_fc


def reduce(
    fn: Callable[[float, float], float], init: float
) -> Callable[[Iterable[float]], float]:
    """Implement a higher-order reduce function."""

    def reduce_fc(array: Iterable[float]) -> float:
        t = init
        for x in array:
            t = fn(t, x)
        return t

    return reduce_fc


def negList(arr: Iterable[float]) -> Iterable[float]:
    """Negate a list of numbers."""
    return map(neg)(arr)


def addLists(arr1: Iterable[float], arr2: Iterable[float]) -> Iterable[float]:
    """Add two lists elementwise."""
    return zipWith(add)(arr1, arr2)


def sum(arr: Iterable[float]) -> float:
    """Sum a list of numbers."""
    return reduce(add, 0)(arr)


def prod(arr: Iterable[float]) -> float:
    """Product of a list of numbers."""
    return reduce(mul, 1)(arr)
