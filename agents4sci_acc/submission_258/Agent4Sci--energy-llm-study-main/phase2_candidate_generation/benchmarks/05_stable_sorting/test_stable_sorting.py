import unittest

class TestStableSorting(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([]), [])

    def test_simple_sort(self):
        data = [(3, "c"), (1, "a"), (2, "b")]
        expected = [(1, "a"), (2, "b"), (3, "c")]
        self.assertEqual(self.solution(data), expected)

    def test_stability(self):
        data = [(2, "a"), (1, "x"), (2, "b")]
        result = self.solution(data)
        self.assertEqual(result[0][1], "x")  # 1 comes first
        self.assertEqual(result[1][1], "a")  # stable order for equal keys
        self.assertEqual(result[2][1], "b")

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.solution([("x", 1)])

