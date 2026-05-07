
from typing import Dict, Tuple, List
from collections import defaultdict

# Bucketing table structure: {hand_class: {texture: idx}}
class CardAbstractionTable:
    def __init__(self):
        self.table = defaultdict(dict)
        self._next = 0

    def get_or_add(self, hand_cls: str, texture: str) -> int:
        if texture not in self.table[hand_cls]:
            self.table[hand_cls][texture] = self._next
            self._next += 1
        return self.table[hand_cls][texture]

    def to_rows(self) -> List[Tuple[int, str, str]]:
        rows = []
        for hc, inner in self.table.items():
            for tex, idx in inner.items():
                rows.append((idx, hc, tex))
        rows.sort()
        return rows
