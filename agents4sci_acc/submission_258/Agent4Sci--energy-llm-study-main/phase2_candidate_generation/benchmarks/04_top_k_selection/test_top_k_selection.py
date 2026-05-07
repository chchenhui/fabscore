import unittest

class TestTopKSelection(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([], 3), [])

    def test_k_zero(self):
        self.assertEqual(self.solution([1, 2, 3], 0), [])

    def test_k_larger_than_list(self):
        result = self.solution([3, 1, 2], 10)
        self.assertEqual(result, [3, 2, 1])

    def test_normal_case(self):
        result = self.solution([5, 1, 3, 9, 2], 3)
        self.assertEqual(result, [9, 5, 3])
