# coding: utf-8
"""
@File        :   infix_to_postfiix_24591.py
@Time        :   2025/03/24 00:40:00
@Author      :   Usercyk
@Description :   Convert infix to postfix
"""
from typing import List


class Solution:
    """
    The solution class
    """

    PRIORITY = {"+": 1, "-": 1, "*": 2, "/": 2}
    NUMBERS = "0123456789."

    def infix_to_postfix(self, infix: str) -> List[str]:
        """
        Conver the infix to postfix

        Arguments:
            infix -- The infix

        Returns:
            The converted postfix
        """
        i = 0
        len_infix = len(infix)
        stack = []
        res = []

        while i < len_infix:
            if infix[i] in self.NUMBERS:
                j = i
                while j < len_infix and infix[j] in self.NUMBERS:
                    j += 1
                num = infix[i:j]
                res.append(num)
                i = j
            elif infix[i] == "(":
                stack.append("(")
                i += 1
            elif infix[i] == ")":
                while stack and stack[-1] != "(":
                    res.append(stack.pop())
                stack.pop()
                i += 1
            else:
                oper = infix[i]
                while stack and stack[-1] != '(' and \
                        self.PRIORITY[oper] <= self.PRIORITY.get(stack[-1], 0):
                    res.append(stack.pop())
                stack.append(oper)
                i += 1
        while stack:
            res.append(stack.pop())
        return res

    def solve(self):
        """
        Solve the problem
        """
        for _ in range(int(input())):
            print(*self.infix_to_postfix(input()))


if __name__ == "__main__":
    Solution().solve()
