# coding: utf-8
"""
@File        :   pre_middle_to_post_22518.py
@Time        :   2025/04/10 15:34:17
@Author      :   Usercyk
@Description :   Convert pre and middle order traversal of a binary tree to post-order traversal.
"""


class Solution:
    """
    The solution class
    """

    def post_order(self, pre_order: str, middle_order: str) -> str:
        """
        Convert pre-order and middle-order traversal of a binary tree to post-order traversal.

        Arguments:
            pre_order -- The pre-order traversal string.
            middle_order -- The middle-order traversal string.

        Returns:
            The post-order traversal string.
        """
        if not pre_order or not middle_order:
            return ""

        root = pre_order[0]

        root_index = middle_order.index(root)

        left_subtree_pre = pre_order[1:1 + root_index]
        left_subtree_middle = middle_order[:root_index]
        right_subtree_pre = pre_order[1 + root_index:]
        right_subtree_middle = middle_order[root_index + 1:]

        return self.post_order(left_subtree_pre, left_subtree_middle) + \
            self.post_order(right_subtree_pre, right_subtree_middle) + root

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                pre_order = input().strip()
                middle_order = input().strip()
                print(self.post_order(pre_order, middle_order))
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()
