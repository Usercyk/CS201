# coding: utf-8
"""
@File        :   reverse_decode_02039.py
@Time        :   2025/03/07 14:08:51
@Author      :   Usercyk
@Description :   Decode
"""
import textwrap


class Solution:
    """
    The solution class
    """

    def solve(self, col: int, message: str):
        """
        Solve the problem
        """
        s = [list(x) for x in textwrap.wrap(message, col)]
        for _ in range(col):
            for idx, val in enumerate(s):
                print(val.pop(-(idx % 2)), end="")


if __name__ == "__main__":
    Solution().solve(int(input()), input())
