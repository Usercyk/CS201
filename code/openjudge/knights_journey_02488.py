# coding: utf-8
"""
@File        :   knights_journey_02488.py
@Time        :   2025/03/07 18:59:38
@Author      :   Usercyk
@Description :   Get the possible paths for a knight to pass every squares in pxq board.
"""


class Solution:
    """
    The solution class
    """
    MOVES = [(-2, -1), (-2, 1), (-1, -2), (-1, 2),
             (1, -2), (1, 2), (2, -1), (2, 1)]

    def __init__(self) -> None:
        self.path = []
        self.p = -1
        self.q = -1
        self.board = []
        self.flag = False

    def travel(self, step: int = 1, x: int = 0, y: int = 0) -> bool:
        """
        Travel the pxq board

        Arguments:
            step -- the current step
            x -- current pos x
            y -- current pos y

        Returns:
            Can the knight travel through all the board
        """
        if step == self.p*self.q:
            self.flag = True
            return True

        for dy, dx in self.MOVES:
            nx, ny = x+dx, y+dy

            if all((not self.flag, 0 <= nx < self.p, 0 <= ny < self.q)):
                if self.board[nx][ny] != 1:
                    self.board[nx][ny] = 1
                    self.path[step] = (nx, ny)
                    self.travel(step+1, nx, ny)
                    self.board[nx][ny] = 0

        return self.flag

    def re_init(self, p: int, q: int):
        """
        Init the board and paths

        Arguments:
            p -- the numbers
            q -- the alphabets
        """
        self.p, self.q = p, q
        self.path = [(0, 0) for _ in range(p*q)]

        self.board = [[0]*(q+1) for _ in range(p+1)]
        self.board[0][0] = 1

        self.flag = False

    def solve(self):
        """
        Solve the problem
        """
        for i in range(int(input())):
            self.re_init(*map(int, input().split()))

            print(f"Scenario #{i+1}:")
            if self.travel():
                ans = (chr(c[1]+ord("A"))+str(c[0]+1) for c in self.path)
                print("".join(ans))
            else:
                print("impossible")
            print("")


if __name__ == "__main__":
    Solution().solve()
