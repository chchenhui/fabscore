from abc import ABC, abstractmethod

class MemoryStorage(ABC):
    @abstractmethod
    async def add(self, memory_item):
        pass

    @abstractmethod
    async def get_all(self):
        pass

    @abstractmethod
    async def delete(self, memory_item):
        pass

    @abstractmethod
    async def query(self, query, top_k=5):
        pass

    @abstractmethod
    async def get_size(self):
        pass    