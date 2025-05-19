# coding: utf-8
"""
@File        :   all_possible_stack_sequence_04077.py
@Time        :   2025/05/16 19:27:54
@Author      :   Usercyk
@Description :   All Possible Stack Sequence
"""


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.n = 0
        self.ans = 0

    def dfs(self, a: int, b: int, step: int) -> None:
        """
        The dfs function

        Arguments:
            a -- Outside number
            b -- Inside number
            step -- Step number
        """
        if step == self.n:
            self.ans += 1
            return
        if a > 0:
            self.dfs(a-1, b+1, step)
        if b > 0:
            self.dfs(a, b-1, step+1)

    def all_possible_stack_sequence(self) -> None:
        """
        The main function
        """
        self.n = int(input())
        self.ans = 0
        self.dfs(self.n, 0, 0)
        print(self.ans)


if __name__ == "__main__":
    Solution().all_possible_stack_sequence()
