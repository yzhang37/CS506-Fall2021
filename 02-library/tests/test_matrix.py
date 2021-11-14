from cs506.matrix import get_determinant
import pytest


@pytest.mark.parametrize('test_input,expected_data', [
    (
        [[4, -8, 6, 2], [-5, 3, -1, 4], [2, -5, 6, 8], [4, 7, 8, 2]], 1900
    ),
    (
        [[10, 20, 32], [2.5, -1.5, 7.5], [2, -3, 18]], -789
    ),
])

def test_matrix(test_input, expected_data):
    print(test_input)
    actual_data = get_determinant(test_input)
    assert int(actual_data) == expected_data
