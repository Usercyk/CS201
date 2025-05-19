# coding: utf-8
"""
@File        :   card_sort_05343.py
@Time        :   2025/05/16 19:33:31
@Author      :   Usercyk
@Description :   Sort the cards
"""
from collections import defaultdict


class Solution:
    """
    The solution class
    """

    def sort_cards(self) -> None:
        """
        The main function
        """
        input()
        cards = input().split()
        d1 = defaultdict(list)
        cards1 = []
        for card in cards:
            d1[card[1]].append(card)
        for k in range(1, 10):
            print(f"Queue{k}:"+" ".join(d1[str(k)]))
            cards1.extend(d1[str(k)])
        d2 = defaultdict(list)
        cards2 = []
        for card in cards1:
            d2[card[0]].append(card)
        for k in ("A", "B", "C", "D"):
            print(f"Queue{k}:"+" ".join(d2[k]))
            cards2.extend(d2[k])
        print(*cards2)


if __name__ == "__main__":
    Solution().sort_cards()
