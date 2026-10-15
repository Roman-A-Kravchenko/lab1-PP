import pytest
from grader_contracts.python_basics import (
    PositiveIntegerInput, TextInput, VectorPairInput,
)
from python_basics import (
    count_vowels, has_unique_characters, count_one_bits,
    multiplicative_persistence, mse, prime_factorization,
    pyramid, is_balanced_number,
)


def test_count_vowels():
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("")) == 0
    assert count_vowels(TextInput("AEIOUaeiou")) == 10
    assert count_vowels(TextInput("bcdfg")) == 0


def test_unique():
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("aA")) is True
    assert has_unique_characters(TextInput("abcda")) is False


def test_one_bits():
    assert count_one_bits(PositiveIntegerInput(0)) == 0
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(255)) == 8
    assert count_one_bits(PositiveIntegerInput(1024)) == 1


def test_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4
    assert multiplicative_persistence(PositiveIntegerInput(10)) == 1


def test_mse():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2, 3], [2, 4, 6])) == pytest.approx(14 / 3)
    assert mse(VectorPairInput([], [])) == 0.0


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(1)) == ""
    assert prime_factorization(PositiveIntegerInput(2)) == "(2)"
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(97)) == "(97)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(5)) == 2
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(2)) == "It is impossible"


def test_balanced():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(1)) is True
    assert is_balanced_number(PositiveIntegerInput(11)) is True
    assert is_balanced_number(PositiveIntegerInput(12)) is True
    assert is_balanced_number(PositiveIntegerInput(121)) is True
    assert is_balanced_number(PositiveIntegerInput(123)) is False