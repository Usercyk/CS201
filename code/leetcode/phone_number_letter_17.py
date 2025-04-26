# coding: utf-8
"""
@File        :   phone_number_letter_17.py
@Time        :   2025/04/26 18:44:33
@Author      :   Usercyk
@Description :   Find all possible letter combinations that the number could represent.
"""
from itertools import product
from typing import List


class Solution:
    """
    The solution class
    """
    MAPPINGS = {
        "2": "abc",
        "3": "def",
        "4": "ghi",
        "5": "jkl",
        "6": "mno",
        "7": "pqrs",
        "8": "tuv",
        "9": "wxyz"
    }

    def letter_combinations(self, digits: str) -> List[str]:
        """
        All possible letter combinations that the number could represent.

        Arguments:
            digits -- 2-9 digit string

        Returns:
            List of all possible letter combinations
        """
        return ["".join(x) for x in product(
            *map(self.MAPPINGS.get, digits))] if digits else []  # type: ignore


print(Solution().letter_combinations("23"))
