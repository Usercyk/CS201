# coding: utf-8
"""
@File        :   cheapest_flight_within_k_stops_787.py
@Time        :   2025/05/24 14:26:21
@Author      :   Usercyk
@Description :   Find the cheapest price for a flight from src to dst with at most k stops.
"""
from typing import List, Optional


class Solution:
    """
    The solution class
    """

    INF = float('inf')

    def find_cheapest_price(self,
                            n: int,
                            flights: List[List[int]],
                            src: int, dst: int, k: int) -> int:
        """
        Find the cheapest price for a flight from src to dst with at most k stops.

        Arguments:
            n -- number of cities
            flights -- list of flights
                flights[i] = [from, to, price]
                from -- the starting city
                to -- the destination city
                price -- the price of the flight
            src -- the starting city
            dst -- the destination city
            k -- the maximum number of stops

        Returns:
            the cheapest price for a flight from src to dst with at most k stops
                if there is no such flight, return -1
        """
        prices = [self.INF] * n
        prices[src] = 0
        predecessors: List[Optional[int]] = [None] * n

        for _ in range(k+1):
            updated = False
            p = prices.copy()
            for u, v, w in flights:
                if prices[u] != self.INF and p[u] + w < prices[v]:
                    prices[v] = p[u] + w
                    predecessors[v] = u
                    updated = True
            if not updated:
                break

        return int(prices[dst]) if prices[dst] != self.INF else -1
