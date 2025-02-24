# coding: utf-8
"""
@File        :   fraction_27653.py
@Time        :   2025/02/23 13:53:51
@Author      :   Usercyk
@Description :   realize the fraction class
"""


class Fraction:
    """
    Fraction class
    """
    @staticmethod
    def gcd(num_1: int, num_2: int) -> int:
        """
        Calculate the greatest common divisor of a and b

        Arguments:
            num_1 -- One of the number
            num_2 -- Another number

        Returns:
            The greatest common divisor of a and b
        """
        if num_2 == 0:
            return num_1
        return Fraction.gcd(num_2, num_1 % num_2)

    def _simplify(self) -> None:
        """
        Simplify the fraction
        """
        g = self.gcd(self.numerator, self.denominator)
        self.numerator //= g
        self.denominator //= g
        if self.denominator < 0:
            self.numerator = -self.numerator
            self.denominator = -self.denominator

    def __init__(self, numerator: int, denominator: int) -> None:
        """
        Initialize the fraction

        Arguments:
            numerator -- The numerator
            denominator -- The denominator
        """
        self.numerator = numerator
        if denominator == 0:
            raise ValueError("Denominator can't be zero")
        self.denominator = denominator
        self._simplify()

    def __str__(self) -> str:
        if self.denominator == 1:
            return str(self.numerator)
        return f"{self.numerator}/{self.denominator}"

    def __add__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator + other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __sub__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator - other.numerator * self.denominator,
                        self.denominator * other.denominator)

    def __mul__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.numerator,
                        self.denominator * other.denominator)

    def __truediv__(self, other: 'Fraction') -> 'Fraction':
        return Fraction(self.numerator * other.denominator,
                        self.denominator * other.numerator)


if __name__ == "__main__":
    a, b, c, d = map(int, input().split())
    print(Fraction(a, b) + Fraction(c, d))
