import uuid
import time


class MemoryItem:
    def __init__(self, agent_id,content, attributes={},embedding=None):
        self.agent_id = agent_id
        self.id = uuid.uuid4()
        self.content = content
        self.timestamp = time.time()
        self.attributes = attributes  # 存储可选的属性字典
        self.embedding= embedding

    def __repr__(self):
        return f"MemoryItem(id={self.id}, content='{self.content}', attributes={self.attributes})"
