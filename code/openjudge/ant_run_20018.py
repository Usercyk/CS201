# coding: utf-8
"""
@File        :   ant_run_20018.py
@Time        :   2025/03/24 13:35:34
@Author      :   Usercyk
@Description :   Calculate the count of ordered pair
"""
from typing import List


class Solution:
    """
    The solution class
    """

    def __init__(self) -> None:
        self.count = 0

    def merge_count(self, sequence: List[int]) -> List[int]:
        """
        Count the ordered pair and sort the sequence

        Arguments:
            sequence -- original sequence
        """
        if len(sequence) <= 1:
            return sequence

        mid = len(sequence)//2
        left_seq = self.merge_count(sequence[:mid])
        right_seq = self.merge_count(sequence[mid:])

        sorted_seq = []
        i, j = 0, 0

        left_len = len(left_seq)
        right_len = len(right_seq)

        while i < left_len and j < right_len:
            if left_seq[i] < right_seq[j]:
                self.count += left_len-i
                sorted_seq.append(right_seq[j])
                j += 1
            else:
                sorted_seq.append(left_seq[i])
                i += 1

        sorted_seq.extend(left_seq[i:])
        sorted_seq.extend(right_seq[j:])

        return sorted_seq

    def solve(self) -> None:
        """
        Solve the problem
        """
        seq = [int(input()) for _ in range(int(input()))]
        self.count = 0
        self.merge_count(seq)
        print(self.count)
        self.count = 0


if __name__ == "__main__":
    Solution().solve()
