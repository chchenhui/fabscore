import unittest

class TestMatrixMultiplication(unittest.TestCase):
    def test_valid_product(self):
        a = [[1, 2], [3, 4]]
        b = [[5, 6], [7, 8]]
        self.assertEqual(self.solution(a, b), [[19, 22], [43, 50]])

    def test_identity_matrix(self):
        a = [[1, 2], [3, 4]]
        i = [[1, 0], [0, 1]]
        self.assertEqual(self.solution(a, i), a)

    def test_dimension_mismatch(self):
        a = [[1, 2, 3]]
        b = [[4, 5], [6, 7]]
        with self.assertRaises(ValueError):
            self.solution(a, b)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.solution([[1, "x"]], [[1], [2]])
