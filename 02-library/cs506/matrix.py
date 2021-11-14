from functools import reduce
import typing


# Takes in a matrix (list of lists)
# and returns the determinant of the matrix.
def get_determinant(matrix: typing.List[typing.List[float]]):
    return determinant_recursive(matrix)


def determinant_recursive(A, total=0):
    indices = list(range(len(A)))

    if len(A) == 2 and len(A[0]) == 2:
        val = A[0][0] * A[1][1] - A[1][0] * A[0][1]
        return val

    for fc in indices: # A) for each focus column, ...
        # find the submatrix ...
        As = [[j for j in i] for i in A] # B) make a copy, and ...
        As = As[1:] # ... C) remove the first row
        height = len(As) # D)

        for i in range(height):
            # E) for each remaining row of submatrix ...
            #     remove the focus column elements
            As[i] = As[i][0:fc] + As[i][fc+1:]

        sign = (-1) ** (fc % 2) # F)
        # G) pass submatrix recursively
        sub_det = determinant_recursive(As)
        # H) total all returns from recursion
        total += sign * A[0][fc] * sub_det

    return total
