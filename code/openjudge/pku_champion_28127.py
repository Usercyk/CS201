# coding: utf-8
"""
@File        :   pku_champion_28127.py
@Time        :   2025/04/06 17:06:25
@Author      :   Usercyk
@Description :   Sort the universities in ICPC
"""


class University:
    """
    The university class
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.questions = {k: False for k in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
        self.update_times = 0

    def update(self, question: str, passed: str) -> None:
        """
        Update the university with the question and passed status

        Arguments:
            question -- The question name
            passed -- The passed status
        """
        self.questions[question] = self.questions[question] or passed == "yes"
        self.update_times += 1

    @property
    def score(self) -> int:
        """
        Calculate the score of the university

        Returns:
            The score of the university
        """
        return sum(1 for k in self.questions if self.questions[k])

    def __str__(self) -> str:
        return f"{self.name} {self.score} {self.update_times}"

    def __lt__(self, other: 'University') -> bool:
        """
        Compare two universities

        Arguments:
            other -- The other university

        Returns:
            True if self is less than other, False otherwise
        """
        if self.score != other.score:
            return self.score > other.score
        if self.update_times != other.update_times:
            return self.update_times < other.update_times
        return self.name < other.name


class ICPC:
    """
    The icpc competition
    """

    def __init__(self) -> None:
        self.universities = {}

    def update(self, submission: str) -> None:
        """
        Update the university with the submission

        Arguments:
            submission -- The submission string
        """
        name, question, passed = submission.split(",")
        if name not in self.universities:
            self.universities[name] = University(name)
        self.universities[name].update(question, passed)

    def run(self) -> None:
        """
        Run the contest
        """
        for _ in range(int(input())):
            submission = input().strip()
            self.update(submission)

        for i, university in enumerate(sorted(self.universities.values())):
            if i >= 12:
                break
            print(f"{i + 1} {university}")


if __name__ == "__main__":
    ICPC().run()
