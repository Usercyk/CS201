# coding: utf-8
"""
@File        :   granpa_is_famous_02092.py
@Time        :   2025/03/07 14:23:07
@Author      :   Usercyk
@Description :   Get the 2nd best player
"""
from typing import List
from collections import defaultdict


class Solution:
    """
    The solution class
    """
    MAX_PLAYER = 10001

    def get_2nd_player(self, players: List[int]) -> List[int]:
        """
        Get the second players

        Arguments:
            players -- all ranked players

        Returns:
            The number of the second player
        """
        rank = defaultdict(int)
        for player in players:
            rank[player] += 1
        invert = defaultdict(list)
        for k, v in rank.items():
            invert[v].append(k)

        second_time = sorted(invert.keys())[-2]
        return sorted(invert[second_time])

    def solve(self):
        """
        Solve the problem
        """
        while True:
            n, m = map(int, input().split())
            if n == m == 0:
                break
            players = []
            for _ in range(n):
                players.extend(list(map(int, input().split())))

            ans = self.get_2nd_player(players)
            print(*ans, end=" \n")


if __name__ == "__main__":
    Solution().solve()
