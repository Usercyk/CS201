# coding: utf-8
"""
@File        :   closest_sum_18156.py
@Time        :   2025/04/06 16:28:46
@Author      :   Usercyk
@Description :   Closest Sum
"""

from typing import List


class Solution:
    """
    The solution class
    """

    def closest_sum(self, nums: List[int], target: int) -> int:
        """
        Find the closest sum to the target.

        Arguments:
            nums -- list of integers
            target -- target integer

        Returns:
            The closest sum to the target.
        """
        nums.sort()
        left, right = 0, len(nums) - 1
        ans = float('inf')
        while left < right:
            current = nums[left] + nums[right]
            if abs(current - target) < abs(ans - target):
                ans = current
            elif abs(current - target) == abs(ans - target):
                ans = min(ans, current)

            if current < target:
                left += 1
            elif current > target:
                right -= 1
            else:
                break
        return int(ans)

    def solve(self) -> None:
        """
        Solve the problem
        """
        target = int(input())
        nums = list(map(int, input().split()))
        print(self.closest_sum(nums, target))


if __name__ == "__main__":
    Solution().solve()
