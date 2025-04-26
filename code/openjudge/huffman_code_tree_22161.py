# coding: utf-8
"""
@File        :   huffman_code_tree_22161.py
@Time        :   2025/04/22 13:52:31
@Author      :   Usercyk
@Description :   Realize a Huffman code tree
"""
from heapq import heapify, heappop, heappush
from typing import List, Optional, Set


class Node:
    """
    Node in the Huffman tree.
    """

    def __init__(self, chars: Set[str], freq: int) -> None:
        self.chars = chars
        self.freq = freq
        self.left: Optional[Node] = None
        self.right: Optional[Node] = None

    def __lt__(self, other: 'Node') -> bool:
        if self.freq != other.freq:
            return self.freq < other.freq
        return min(self.chars) < min(other.chars)


class HuffmanTree:
    """
    Huffman tree.
    """

    def __init__(self, chars: List[str], freqs: List[int]) -> None:
        self.chars = chars
        self.freqs = freqs
        self.root = None
        self.code_map = {}
        self.build_tree()
        self.build_code_map(self.root, "")

    def build_tree(self) -> None:
        """
        Build the Huffman tree from the characters and their frequencies.
        """
        nodes = [Node({char}, freq)
                 for char, freq in zip(self.chars, self.freqs)]
        heapify(nodes)
        while len(nodes) > 1:
            nodes.sort()
            left = heappop(nodes)
            right = heappop(nodes)
            merged = Node(left.chars.union(right.chars),
                          left.freq + right.freq)
            merged.left = left
            merged.right = right
            heappush(nodes, merged)
        self.root = nodes[0]

    def build_code_map(self, node: Optional[Node], path: str) -> None:
        """
        Build the code map for the characters in the Huffman tree.

        Arguments:
            node -- node to traverse
            path -- the path to the current node
        """
        if node is None:
            return
        if not node.left and not node.right:  # Leaf node
            self.code_map[next(iter(node.chars))] = path
            return
        self.build_code_map(node.left, path + "0")
        self.build_code_map(node.right, path + "1")

    def encode(self, string: str) -> str:
        """
        Encode the string using the Huffman tree.

        Arguments:
            string -- the string to encode

        Returns:
            The encoded string.
        """
        encoded_string = "".join(self.code_map[char] for char in string)
        return encoded_string

    def decode(self, string: str) -> str:
        """
        Decode the string using the Huffman tree.

        Arguments:
            string -- the string to decode

        Returns:
            The decoded string.
        """
        decoded_string = ""
        node = self.root
        for bit in string:
            if node is None:
                raise ValueError("Invalid encoded string")
            if bit == "0":
                node = node.left
            else:
                node = node.right
            if node is None:
                raise ValueError("Invalid encoded string")
            if not node.left and not node.right:
                decoded_string += next(iter(node.chars))
                node = self.root
        return decoded_string

    @classmethod
    def from_input_string(cls) -> 'HuffmanTree':
        """
        Create a Huffman tree from the input string.
        """
        chars = []
        freqs = []
        for _ in range(int(input())):
            c, f = input().split()
            f = int(f)
            chars.append(c)
            freqs.append(f)
        return cls(chars, freqs)


if __name__ == "__main__":
    huffman = HuffmanTree.from_input_string()
    while True:
        try:
            s = input()
            if "1" in s or "0" in s:
                print(huffman.decode(s))
            else:
                print(huffman.encode(s))
        except EOFError:
            break
