# coding: utf-8
"""
@File        :   minimum_award_22508.py
@Time        :   2025/05/24 14:42:18
@Author      :   Usercyk
@Description :   Calculate the minimum award
"""
from typing import List, Tuple


class Vertex:
    """
    The vertex
    """

    def __init__(self, tag: int) -> None:
        self.tag: int = tag
        self.in_degree: int = 0
        self.out_degree: int = 0
        self.out_adjacent: List["Vertex"] = []

    def connect_to(self, other: "Vertex"):
        """
        Connect this vertex to another vertex

        Arguments:
            other -- The other vertex

        Returns:
            The other vertex
        """
        self.out_adjacent.append(other)
        self.out_degree += 1
        other.in_degree += 1


class Graph:
    """
    The graph
    """

    def __init__(self, n: int) -> None:
        self._vertices: List[Vertex] = [Vertex(i+1) for i in range(n)]

    def connect(self, u: int, v: int) -> None:
        """
        Connect u to v

        Arguments:
            u -- The first vertex
            v -- The second vertex
        """
        self._vertices[u].connect_to(self._vertices[v])

    def minimum_award(self) -> int:
        """
        Calculate the minimum award

        Returns:
            The minimum award
        """
        queue: List[Tuple[Vertex, int]] = [
            (v, 100) for v in self._vertices if v.in_degree == 0]
        award = 0
        while queue:
            vertex, i = queue.pop(0)
            award += i
            for adj in vertex.out_adjacent:
                adj.in_degree -= 1
                if adj.in_degree == 0:
                    queue.append((adj, i+1))
        return award


class Solution:
    """
    The solution class
    """

    def solve(self) -> None:
        """
        Solve the problem
        """
        v, a = map(int, input().split())
        graph = Graph(v)
        for _ in range(a):
            v, u = map(int, input().split())
            graph.connect(u, v)
        print(graph.minimum_award())


if __name__ == "__main__":
    Solution().solve()
