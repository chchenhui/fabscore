import unittest

class TestArraySum(unittest.TestCase):
    def test_empty_list(self):
        self.assertEqual(self.solution([]), 0)

    def test_small_numbers(self):
        self.assertEqual(self.solution([1, 2, 3]), 6)

    def test_floats(self):
        self.assertAlmostEqual(self.solution([0.5, 1.5, 2.0]), 4.0)

    def test_type_error(self):
        with self.assertRaises(TypeError):
            self.solution([1, "two", 3])
