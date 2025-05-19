# coding: utf-8
"""
@File        :   road_07735.py
@Time        :   2025/05/16 20:12:04
@Author      :   Usercyk
@Description :   Dijkstra to find the shortest path
"""
from heapq import heappop, heappush


class Solution:
    """
    The solution class
    """
    INF = float("inf")

    def solve(self) -> None:
        """
        Solve the problem
        """
        k = int(input())
        n = int(input())
        r = int(input())

        adjacency_matrix = [[] for _ in range(n + 1)]
        for _ in range(r):
            s, d, l, t = map(int, input().split())
            adjacency_matrix[s].append((d, l, t))

        distances = [[self.INF] * (k + 1) for _ in range(n + 1)]
        distances[1][0] = 0

        heap = [(0, 1, 0)]
        flag = False

        while heap:
            current_distance, current_city, current_cost = heappop(heap)
            if current_city == n:
                print(current_distance)
                flag = True
                break
            if current_distance > distances[current_city][current_cost]:
                continue
            for next_city, length, cost in adjacency_matrix[current_city]:
                new_cost = current_cost + cost
                if new_cost > k:
                    continue
                new_distance = current_distance + length
                if new_distance < distances[next_city][new_cost]:
                    distances[next_city][new_cost] = new_distance
                    heappush(heap, (new_distance, next_city, new_cost))

        if not flag:
            print("-1")


if __name__ == "__main__":
    Solution().solve()
