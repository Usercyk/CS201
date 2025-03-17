# coding: utf-8
"""
@File        :   chemistry_article_20140.py
@Time        :   2025/03/17 13:59:38
@Author      :   Usercyk
@Description :   Decode the multiple run length encoding
"""


class Solution:
    """
    The solution
    """

    def decode(self, code: str):
        """
        Decode the mutiple RLE
        """
        stack = []
        curr_num, curr_str = 0, ""

        for c in code:
            if c.isdigit():
                curr_num = curr_num*10+int(c)
            elif c == "[":
                stack.append((curr_num, curr_str))
                curr_num, curr_str = 0, ""
            elif c == "]":
                prev_num, prev_str = stack.pop()
                curr_str = prev_str+curr_str*curr_num
                curr_num = prev_num
            else:
                curr_str += c

        return curr_str


if __name__ == "__main__":
    print(Solution().decode(input()))
