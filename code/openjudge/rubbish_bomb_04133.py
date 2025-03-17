# coding: utf-8
"""
@File        :   rubbish_bomb_04133.py
@Time        :   2025/03/07 13:55:11
@Author      :   Usercyk
@Description :   Rubbish bomb
"""


from itertools import product
from typing import List, Tuple


class Solution:
    """
    The solution class
    """

    def explode(self, d: int, rubbishes: List[Tuple[int, int, int]], bomb_cord: Tuple[int, int]) -> int:
        """
        Explode the bomb

        Arguments:
            d -- The power of the bomb
            rubbishes -- The rubbish in the street. (x, y, rubbish_cnt)
            bomb_cord -- The place where the bomb explode. (bomb_x, bomb_y)

        Returns:
            The count of cleared rubbish
        """
        bomb_x, bomb_y = bomb_cord
        res = 0
        for rubbish_x, rubbish_y, rubbish_cnt in rubbishes:
            if bomb_x-d <= rubbish_x <= bomb_x+d and bomb_y-d <= rubbish_y <= bomb_y+d:
                res += rubbish_cnt
        return res

    def solve(self):
        """
        solve the problem
        """
        d = int(input())
        rubbishes = []
        for _ in range(int(input())):
            rubbishes.append(tuple(map(int, input().split())))
        ans = -1
        cnt = 0
        for p in product(range(1025), range(1025)):
            res = self.explode(d, rubbishes, p)
            if res > ans:
                ans = res
                cnt = 1
            elif res == ans:
                cnt += 1
        print(cnt, ans)


if __name__ == "__main__":
    Solution().solve()
