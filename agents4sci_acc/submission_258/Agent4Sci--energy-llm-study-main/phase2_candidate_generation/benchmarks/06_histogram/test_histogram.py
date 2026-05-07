import unittest

class TestHistogram(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([]), {})

    def test_counts_numbers(self):
        data = [1, 2, 2, 3, 3, 3]
        expected = {1: 1, 2: 2, 3: 3}
        self.assertEqual(self.solution(data), expected)

    def test_counts_strings(self):
        data = ["a", "b", "a"]
        expected = {"a": 2, "b": 1}
        self.assertEqual(self.solution(data), expected)

