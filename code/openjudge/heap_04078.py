# coding: utf-8
"""
@File        :   heap_04078.py
@Time        :   2025/04/22 12:42:24
@Author      :   Usercyk
@Description :   Heap
"""
from typing import List


class Heap:
    """
    A simple implementation of a heap.
    """

    def __init__(self, key=None) -> None:
        self._array: List[int] = []
        self._key = key if key is not None else lambda x: x
        self._size = 0

    def left(self, i: int) -> int:
        """
        Returns the index of the left child of the node at index i.
        """
        return 2 * i + 1

    def right(self, i: int) -> int:
        """
        Returns the index of the right child of the node at index i.
        """
        return 2 * i + 2

    def parent(self, i: int) -> int:
        """
        Returns the index of the parent of the node at index i.
        """
        if i == 0:
            raise IndexError("Root has no parent")
        return (i - 1) // 2

    def insert(self, value: int) -> None:
        """
        Inserts a new value into the heap.
        """
        self._array.append(value)
        self._size += 1
        self._sift_up(self._size - 1)

    def heappop(self) -> int:
        """
        Removes and returns the smallest value from the heap.
        """
        if self._size == 0:
            raise IndexError("Heap is empty")
        root = self._array[0]
        last_element = self._array.pop()
        self._size -= 1
        if self._size > 0:
            self._array[0] = last_element
            self._sift_down(0)
        return root

    def _cmp(self, i: int, j: int) -> bool:
        """
        Compares the values at indices i and j in the heap.
        """
        return self._key(self._array[i]) < self._key(self._array[j])

    def _sift_up(self, i: int) -> None:
        """
        Moves the node at index i up to its correct position in the heap.
        """
        while i > 0 and self._cmp(i, self.parent(i)):
            self._array[i], self._array[self.parent(
                i)] = self._array[self.parent(i)], self._array[i]
            i = self.parent(i)

    def _sift_down(self, i: int) -> None:
        """
        Moves the node at index i down to its correct position in the heap.
        """
        while True:
            left = self.left(i)
            right = self.right(i)
            smallest = i

            if left < self._size and self._cmp(left, smallest):
                smallest = left
            if right < self._size and self._cmp(right, smallest):
                smallest = right
            if smallest == i:
                break
            self._array[i], self._array[smallest] = self._array[smallest], self._array[i]
            i = smallest

    def __len__(self) -> int:
        """
        Returns the number of elements in the heap.
        """
        return self._size

    def to_list(self) -> List[int]:
        """
        Returns the elements of the heap as a list.
        """
        return self._array[:self._size]

    def __str__(self) -> str:
        """
        Returns a string representation of the heap.
        """
        return str(self.to_list())


class Manager:
    """
    A simple manager for the heap.
    """

    def __init__(self, key=None) -> None:
        self._heap = Heap(key)

    def work(self, op: str) -> None:
        """
        Deal with the heap operations based on the type and value provided.

        Arguments:
            op -- typ val
            typ -- 1: insert, 2: pop
            val -- the value to be inserted or ignored if type is 2
        """
        typ = int(op[0])
        if typ == 1:
            val = int(op[2:])
            self._heap.insert(val)
        elif typ == 2:
            if len(self._heap) > 0:
                print(self._heap.heappop())
            else:
                raise IndexError("Heap is empty")
        else:
            raise ValueError("Invalid operation type")


if __name__ == "__main__":
    manager = Manager()
    for _ in range(int(input())):
        manager.work(input().strip())
