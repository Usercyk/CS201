# coding: utf-8
"""
@File        :   treasure_binary_tree_24637.py
@Time        :   2025/05/16 20:32:27
@Author      :   Usercyk
@Description :   Binary tree treasure
"""
from typing import List


class Solution:
    """
    The solution class
    """

    def dp(self, tree: List[int], n: int, idx: int) -> int:
        """
        DP function to find the maximum treasure
        """
        if idx > n:
            return 0
        l = self.dp(tree, n, idx*2)
        r = self.dp(tree, n, idx*2+1)
        ll = self.dp(tree, n, idx*4)
        lr = self.dp(tree, n, idx*4+1)
        rl = self.dp(tree, n, idx*4+2)
        rr = self.dp(tree, n, idx*4+3)
        return max(tree[idx]+ll+lr+rl+rr, l+r)

    def solve(self) -> None:
        """
        Solve the problem
        """
        n = int(input())
        tree = [0, *map(int, input().split())]
        result = self.dp(tree, n, 1)
        print(result)


if __name__ == "__main__":
    Solution().solve()
