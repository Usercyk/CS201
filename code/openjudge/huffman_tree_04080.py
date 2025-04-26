# coding: utf-8
"""
@File        :   huffman_tree_04080.py
@Time        :   2025/04/21 20:10:34
@Author      :   Usercyk
@Description :   Build a Huffman tree from a list of weights
"""
import heapq
from typing import List


class Solution:
    """
    The solution class
    """

    def min_weighted_path_length(self, weights: List[int]) -> int:
        """
        Calculate the minimum weighted path length of a Huffman tree.

        Arguments:
            weights -- a list of weights

        Returns:
            The minimum weighted path length of the Huffman tree
        """
        heap = weights.copy()
        heapq.heapify(heap)
        ans = 0

        while len(heap) > 1:
            w1, w2 = heapq.heappop(heap), heapq.heappop(heap)
            ans += w1+w2
            heapq.heappush(heap, w1+w2)

        return ans


if __name__ == "__main__":
    input()
    print(Solution().min_weighted_path_length(list(map(int, input().split()))))
