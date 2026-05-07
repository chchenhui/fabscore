import unittest

class TestCSVETL(unittest.TestCase):
    def test_valid_csv(self):
        csv_str = "id,name\n1,Alice\n2,Bob"
        result = self.solution(csv_str)
        expected = [{"id": "1", "name": "Alice"}, {"id": "2", "name": "Bob"}]
        self.assertEqual(result, expected)

    def test_empty_csv(self):
        csv_str = ""
        with self.assertRaises(ValueError):
            self.solution(csv_str)

    def test_invalid_csv(self):
        csv_str = "id,name\n1"
        with self.assertRaises(ValueError):
            self.solution(csv_str)
