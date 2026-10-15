"""Заготовки задач на базовом Python."""

from grader_contracts.python_basics import (
    PositiveIntegerInput, TextInput, VectorPairInput,
)


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiouAEIOU")
    return sum(1 for ch in text if ch in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        product = 1
        for digit in str(number):
            product *= int(digit)
        number = product
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    n = len(predicted)
    if n == 0:
        return 0.0
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / n


def prime_factorization(data: PositiveIntegerInput) -> str:
    n = data.value
    result = []
    divisor = 2
    while divisor * divisor <= n:
        if n % divisor == 0:
            power = 0
            while n % divisor == 0:
                n //= divisor
                power += 1
            result.append(f"({divisor}**{power})" if power > 1 else f"({divisor})")
        divisor += 1
    if n > 1:
        result.append(f"({n})")
    return "".join(result)


def pyramid(data: PositiveIntegerInput) -> int | str:
    n = data.value
    total, k = 0, 0
    while total < n:
        k += 1
        total += k * k
    return k if total == n else "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    digits = [int(c) for c in str(data.value)]
    n = len(digits)
    mid = n // 2
    if n % 2 == 1:
        left, right = digits[:mid], digits[mid + 1:]
    else:
        left, right = digits[:mid - 1], digits[mid + 1:]
    return sum(left) == sum(right)
