import unittest

class TestStringDeduplication(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([]), [])

    def test_no_duplicates(self):
        data = ["a", "b", "c"]
        self.assertEqual(self.solution(data), data)

    def test_with_duplicates(self):
        data = ["a", "b", "a", "c", "b"]
        self.assertEqual(self.solution(data), ["a", "b", "c"])

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.solution(["a", 1, "b"])
