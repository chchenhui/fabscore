import unittest

class TestPrefixSum(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([]), [])

    def test_single_element(self):
        self.assertEqual(self.solution([5]), [5])

    def test_multiple_elements(self):
        self.assertEqual(self.solution([1, 2, 3, 4]), [1, 3, 6, 10])

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.solution([1, "x", 3])

