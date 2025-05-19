# coding: utf-8
"""
@File        :   main.py
@Time        :   2025/02/23 13:07:24
@Author      :   Usercyk
@Description :   test
"""


def is_satisfied(scores, a):
    adjusted_scores = [a * x + (1.1 ** (a * x)) for x in scores]
    excellent_count = sum(score >= 85 for score in adjusted_scores)
    return excellent_count / len(scores) >= 0.6


def find_min_b_binary_search(scores):
    low, high = 1, 10**9

    while low < high:
        mid = (low + high) // 2
        a = mid / 1000000000

        if is_satisfied(scores, a):
            high = mid  # 尝试更小的 b
        else:
            low = mid + 1  # 需要更大的 b

    return low


# 示例输入
scores = [50.5, 100.0, 40.0]
result = find_min_b_binary_search(scores)
print(result)
