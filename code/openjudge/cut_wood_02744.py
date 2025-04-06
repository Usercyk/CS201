# coding: utf-8
"""
@File        :   cut_wood_02744.py
@Time        :   2025/04/06 15:52:26
@Author      :   Usercyk
@Description :   Cut the wood
"""
from typing import List


class Solution:
    """
    The solution
    """

    def check(self, length: int, woods: List[int], k: int) -> bool:
        """
        Check if the wood can be cut

        Arguments:
            length -- The length of wood
            woods -- The length of woods
            k -- The number of pieces

        Returns:
            True if the wood can be cut, False otherwise
        """
        return sum(wood // length for wood in woods) >= k

    def cut_wood(self, k: int, woods: List[int]) -> int:
        """
        Cut the wood

        Arguments:
            k -- The number of pieces
            woods -- The length of woods

        Returns:
            The max length of wood
        """
        left, right = 1, max(woods)
        result = 0
        while left <= right:
            mid = (left + right) // 2
            if self.check(mid, woods, k):
                result = mid
                left = mid + 1
            else:
                right = mid - 1
        return result


if __name__ == "__main__":
    N, K = map(int, input().split())
    Woods = [int(input()) for _ in range(N)]
    print(Solution().cut_wood(K, Woods))
