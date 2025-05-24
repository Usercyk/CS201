# coding: utf-8
"""
@File        :   agri-net_01258.py
@Time        :   2025/05/23 20:10:42
@Author      :   Usercyk
@Description :   Build Agri-Net
"""
from itertools import islice
from typing import Iterable, List
import heapq


class Solution:
    """
    The solution class
    """

    def minimum_length(self, distances: List[List[int]], n: int) -> int:
        """
        Calculate the minimum length of the distances

        Arguments:
            distances -- the distances between the farms
            n -- the number of farms

        Returns:
            The minimum length of the MST
        """
        visited = [False] * n
        min_edge = [float('inf')] * n
        min_edge[0] = 0
        heap = [(0, 0)]  # (cost, node)
        total = 0

        while heap:
            cost, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True
            total += cost
            for v in range(n):
                if not visited[v] and distances[u][v] < min_edge[v]:
                    min_edge[v] = distances[u][v]
                    heapq.heappush(heap, (distances[u][v], v))
        return total

    def chunked_iter(self, iterable: Iterable, n: int) -> Iterable[List[int]]:
        """
        Chunk an iterable into chunks of size n

        Arguments:
            iterable -- the iterable to be chunked
            n -- the size of each chunk

        Returns:
            An iterable of chunks
        """
        it = iter(iterable)
        while True:
            chunk = list(islice(it, n))
            if not chunk:
                break
            yield chunk

    def solve(self) -> None:
        """
        Solve the problem
        """
        while True:
            try:
                n = int(input())
                lines = []
                while len(lines) < n * n:
                    line = map(int, input().split())
                    lines.extend(line)
                it = iter(lines)
                distances = list(self.chunked_iter(it, n))
                print(self.minimum_length(distances, n))
            except EOFError:
                break


if __name__ == "__main__":
    Solution().solve()
