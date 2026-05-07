import unittest

class TestBFS(unittest.TestCase):
    def test_simple_graph(self):
        graph = {"A": ["B", "C"], "B": ["D"], "C": [], "D": []}
        result = self.solution(graph, "A")
        self.assertEqual(result, ["A", "B", "C", "D"])

    def test_empty_graph(self):
        result = self.solution({}, "A")
        self.assertEqual(result, [])

    def test_missing_start_node(self):
        graph = {"A": ["B"]}
        result = self.solution(graph, "X")
        self.assertEqual(result, [])

