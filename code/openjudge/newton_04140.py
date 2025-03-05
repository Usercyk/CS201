# coding: utf-8
"""
@File        :   newton_04140.py
@Time        :   2025/03/01 16:14:57
@Author      :   Usercyk
@Description :   Using Newton's method to solve the equation f(x) = 0
"""


class Solution:
    """
    The solution
    """

    def phi(self, x: float) -> float:
        """
        The iteration function

        Arguments:
            x -- The nth x value

        Returns:
            The (n+1)th x value
        """
        return (2*x**3-5*x**2+80)/(3*x**2-10*x+10)

    def solve(self, x_init: float = 5.0, eps: float = 1e-15) -> float:
        """
        Solve the equation f(x) = x**3-5*x**2+10*x-80 = 0

        Keyword Arguments:
            x_init -- The initial value of x (default: {5.0})
            eps -- The precision (default: {1e-15})

        Returns:
            The solution of the equation
        """
        x = x_init
        while True:
            x_next = self.phi(x)
            if abs(x_next-x) < eps:
                return x_next
            x = x_next


if __name__ == "__main__":
    print(f"{Solution().solve():.9f}")
