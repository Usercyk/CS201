# coding: utf-8
"""
@File        :   design_browser_history_1472.py
@Time        :   2025/03/23 23:40:36
@Author      :   Usercyk
@Description :   Design a browser history
"""
from typing import Optional


class DoubleListNode:
    """
    Double list node
    """

    def __init__(self, url: str) -> None:
        self.url = url
        self._prior: Optional[DoubleListNode] = None
        self._next: Optional[DoubleListNode] = None

    @property
    def prior(self) -> Optional["DoubleListNode"]:
        """
        _prior getter

        Returns:
            self._prior
        """
        return self._prior

    @property
    def next(self) -> Optional["DoubleListNode"]:
        """
        _next getter

        Returns:
            self._next
        """
        return self._next

    @next.setter
    def next(self, other: Optional["DoubleListNode"]) -> None:
        self._next = other
        if other is not None:
            other.set_prior(self)

    @prior.setter
    def prior(self, other: Optional["DoubleListNode"]) -> None:
        self._prior = other
        if other is not None:
            other.set_next(self)

    def set_next(self, other: Optional["DoubleListNode"]) -> None:
        """
        Set next avoid infinite recursion

        Arguments:
            other -- the next node
        """
        self._next = other

    def set_prior(self, other: Optional["DoubleListNode"]) -> None:
        """
        Set prior avoid infinite recursion

        Arguments:
            other -- the prior node
        """
        self._prior = other


class BrowserHistory:
    """
    The browser history
    """

    def __init__(self, homepage: str):
        """
        Initialize the browser history

        Arguments:
            homepage -- the homepage
        """
        self.current = DoubleListNode(homepage)

    def visit(self, url: str) -> None:
        """
        Visit the url, clear the forward history

        Arguments:
            url -- the url
        """
        v = DoubleListNode(url)
        self.current.next = v
        self.current = v

    def back(self, steps: int) -> str:
        """
        Go back steps in history, return the url.

        Arguments:
            steps -- Up to steps back

        Returns:
            the url
        """
        while self.current.prior is not None and steps:
            self.current = self.current.prior
            steps -= 1

        return self.current.url

    def forward(self, steps: int) -> str:
        """
        Go forward steps in history, return the url.

        Arguments:
            steps -- Up to steps forward

        Returns:
            the url
        """
        while self.current.next is not None and steps:
            self.current = self.current.next
            steps -= 1

        return self.current.url
