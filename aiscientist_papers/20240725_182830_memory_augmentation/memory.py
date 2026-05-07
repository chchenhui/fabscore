import torch
import torch.nn as nn

class ExternalMemory(nn.Module):
    def __init__(self, memory_size, embedding_dim):
        super(ExternalMemory, self).__init__()
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.memory = nn.Parameter(torch.randn(memory_size, embedding_dim))

    def read(self, indices):
        return self.memory[indices]

    def write(self, indices, values):
        self.memory.data[indices] = values
