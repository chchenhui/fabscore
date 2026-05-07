import unittest
import json

class TestJSONParsingFiltering(unittest.TestCase):
    def test_valid_json(self):
        data = '[{"active": true}, {"active": false}, {"active": true}]'
        result = self.solution(data)
        self.assertEqual(result, [{"active": True}, {"active": True}])

    def test_invalid_json(self):
        with self.assertRaises(ValueError):
            self.solution('not a json')

    def test_missing_active_key(self):
        data = '[{"x": 1}, {"active": true}]'
        result = self.solution(data)
        self.assertEqual(result, [{"active": True}])
