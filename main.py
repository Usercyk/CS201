from itertools import islice


def chunked_iterable(iterable, size):
    return iter(lambda: tuple(islice(iter(iterable), size)), ())


lst = [1, 2, 3, 4, 5, 6, 7, 8, 9]


result = list(chunked_iterable(lst, 3))


print(result)  # 输出: [(1, 2, 3), (4, 5, 6), (7, 8, 9)]
