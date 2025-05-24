# coding: utf-8
"""
@File        :   quadratic_probing_17975.py
@Time        :   2025/05/23 19:49:35
@Author      :   Usercyk
@Description :   Quadratic Probing
"""
import sys
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    def build_hash_table(self, m: int, num_list: List[int]) -> List[int]:
        """
        Build a hash table using quadratic probing

        Arguments:
            m -- the size of the hash table
            num_list -- the list of elements to be inserted

        Returns:
            A list representing the hash table
        """
        hash_table: List[Optional[int]] = [None] * m
        res = []
        for num in num_list:
            i = 0
            while True:
                index = (num + i * i) % m
                if hash_table[index] is None or hash_table[index] == num:
                    hash_table[index] = num
                    res.append(index)
                    break
                index = (num - i * i) % m
                if hash_table[index] is None or hash_table[index] == num:
                    hash_table[index] = num
                    res.append(index)
                    break
                i += 1
        return res

    def solve(self) -> None:
        """
        Solve the problem
        """
        data = sys.stdin.read().split()
        index = 0
        n = int(data[index])
        index += 1
        m = int(data[index])
        index += 1
        num_list = [int(i) for i in data[index:index+n]]

        print(*self.build_hash_table(m, num_list))


if __name__ == "__main__":
    Solution().solve()
