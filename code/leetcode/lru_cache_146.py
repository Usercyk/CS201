# coding: utf-8
"""
@File        :   lru_cache_146.py
@Time        :   2025/03/27 15:54:45
@Author      :   Usercyk
@Description :   LRU Cache
"""


from typing import Dict, Optional


class DoubleLinkedNode:
    """
    double linked node
    """

    def __init__(self, key: int = -1, value: int = -1) -> None:
        self.key: int = key
        self.value: int = value
        self.prior: Optional[DoubleLinkedNode] = None
        self.next: Optional[DoubleLinkedNode] = None

    def set_next(self, other: Optional["DoubleLinkedNode"]) -> None:
        """
        Link two nodes

        Arguments:
            other -- Another node
        """
        if isinstance(other, DoubleLinkedNode):
            self.next = other
            other.prior = self
        if other is None:
            self.next = None

    def set_prior(self, other: Optional["DoubleLinkedNode"]) -> None:
        """
        Link two nodes

        Arguments:
            other -- Another node
        """
        if isinstance(other, DoubleLinkedNode):
            self.prior = other
            other.next = self
        if other is None:
            self.prior = None


class LRUCache:
    """
    The LRU cache
    """

    def __init__(self, capacity: int):
        self.capacity: int = capacity

        self.cache: Dict[int, DoubleLinkedNode] = dict()
        self.size: int = 0

        self.head: DoubleLinkedNode = DoubleLinkedNode()
        self.tail: DoubleLinkedNode = DoubleLinkedNode()
        self.head.set_next(self.tail)

    def get(self, key: int) -> int:
        """
        Get value in O(1)

        Arguments:
            key -- The key

        Returns:
            The value of the key
        """
        if key not in self.cache:
            return -1
        node = self.cache[key]
        assert node.prior is not None
        node.prior.set_next(node.next)
        node.set_next(self.head.next)
        self.head.set_next(node)
        return node.value

    def put(self, key: int, value: int) -> None:
        """
        Put items to the cache in O(1)

        Arguments:
            key -- The key
            value -- The value
        """
        if key in self.cache:
            node = self.cache[key]
            node.value = value

            assert node.prior is not None
            node.prior.set_next(node.next)
            node.set_next(self.head.next)
            self.head.set_next(node)
            return

        node = DoubleLinkedNode(key, value)
        self.cache[key] = node

        node.set_next(self.head.next)
        self.head.set_next(node)

        self.size += 1

        if self.size > self.capacity:
            node = self.tail.prior
            assert node is not None
            assert node.prior is not None
            node.prior.set_next(self.tail)

            self.cache.pop(node.key)
            self.size -= 1
