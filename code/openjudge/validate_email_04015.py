# coding: utf-8
"""
@File        :   validate_email_04015.py
@Time        :   2025/03/07 13:09:45
@Author      :   Usercyk
@Description :   validate email
"""


class Solution:
    """
    The solution class
    """

    def validate(self, s: str) -> bool:
        """
        Validate email

        Arguments:
            s -- The email string

        Returns:
            True if the email is valid, False otherwise
        """
        split_email = s.strip().split('@')
        if len(split_email) != 2:
            return False
        head, tail = split_email
        if head == '' or tail == '':
            return False
        if head[0] == '.' or head[-1] == '.' or tail[0] == "." or tail[-1] == ".":
            return False
        return "." in tail

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                email = input()
                print("YES" if self.validate(email) else "NO")
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()
