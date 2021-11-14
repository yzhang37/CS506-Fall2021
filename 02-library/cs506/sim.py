import math


def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)


def manhattan_dist(x, y):
    res = 0.0
    for i in range(len(x)):
        res += abs(x[i] - y[i])
    return res


def jaccard_dist(x, y):
    if len(x) == 0 or len(y) == 0:
        raise ValueError("lengths must not be zero")

    large_size = max(len(x), len(y))

    # padding to the same size
    new_x = x + [0] * (large_size - len(x))
    new_y = y + [0] * (large_size - len(y))

    intersect = sum([1 if i == j else 0 for i, j in zip(new_x, new_y)])

    return 1 - intersect / large_size


def cosine_sim(x, y):
    # both two vector must have value.
    if len(x) == 0 or len(y) == 0:
        raise ValueError("lengths must not be zero")
    elif len(x) != len(y):
        raise ValueError("lengths must be equal")

    sum_ab = 0.0
    sum_a2 = 0.0
    sum_b2 = 0.0

    for a, b in zip(x, y):
        sum_ab += a * b
        sum_a2 += a * a
        sum_b2 += b * b

    if sum_a2 == 0.0 or sum_b2 == 0.0:
        raise ValueError("vector should should not be all 0 values.")

    return sum_ab / (math.sqrt(sum_a2) * math.sqrt(sum_b2))

