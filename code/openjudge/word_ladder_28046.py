# coding: utf-8
"""
@File        :   word_ladder_28046.py
@Time        :   2025/04/26 19:30:43
@Author      :   Usercyk
@Description :   Find the shortest transformation sequence from beginWord to endWord.
"""
from collections import defaultdict, deque
from typing import Dict, List, Optional


class Solution:
    """
    The solution class
    """

    def build_graph(self, word_list: List[str]) -> Dict[str, List[str]]:
        """
        Build a graph from the word list.

        Arguments:
            word_list -- the list of words

        Returns:
            A dictionary representing the graph
        """
        graph = defaultdict(list)
        pattern_map = defaultdict(list)
        for word in word_list:
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                pattern_map[pattern].append(word)
        for word in word_list:
            neighbors = set()
            for i in range(len(word)):
                pattern = word[:i] + '*' + word[i+1:]
                neighbors.update(pattern_map[pattern])
            neighbors.discard(word)
            graph[word] = list(neighbors)
        return graph

    def find_shortest_transformation(self,
                                     begin_word: str,
                                     end_word: str,
                                     graph: Dict[str, List[str]]) -> Optional[List[str]]:
        """
        Find the shortest transformation sequence from begin_word to end_word.

        Arguments:
            begin_word -- The starting word
            end_word -- The ending word
            graph -- The graph built from the word list

        Returns:
            The path from begin_word to end_word, or an empty list if no path exists
        """
        queue = deque([(begin_word, [begin_word])])
        visited = {begin_word}
        while queue:
            current_word, path = queue.popleft()
            if current_word == end_word:
                return path
            for neighbor in graph[current_word]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return None

    def solve(self) -> None:
        """
        Solve the problem
        """
        word_list = [input().strip() for _ in range(int(input()))]
        begin_word, end_word = input().split()

        if begin_word not in word_list or end_word not in word_list:
            print("NO")
            return

        graph = self.build_graph(word_list + [begin_word, end_word])

        s = self.find_shortest_transformation(begin_word, end_word, graph)
        if s is None:
            print("NO")
        else:
            print(*s)


if __name__ == '__main__':
    Solution().solve()
