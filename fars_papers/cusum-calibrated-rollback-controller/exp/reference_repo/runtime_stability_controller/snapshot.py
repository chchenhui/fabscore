"""
Snapshot and recovery utilities.

Responsible for saving and restoring model and optimizer state
to enable exact rollback.
"""

import copy


class SnapshotManager:
    """
    Interface for saving and restoring safe-state snapshots.
    """

    def save(self, model, optimizer):
        raise NotImplementedError

    def restore(self, model, optimizer):
        raise NotImplementedError


class InMemorySnapshotManager(SnapshotManager):
    """
    Simple snapshot manager that stores model + optimizer state in memory.
    """

    def __init__(self):
        self._model_state = None
        self._optimizer_state = None

    def save(self, model, optimizer):
        # Clone model parameters/buffers to ensure independence from future updates
        self._model_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        # Optimizer state can contain nested tensors; deepcopy is the simplest correct baseline
        self._optimizer_state = copy.deepcopy(optimizer.state_dict())

    def restore(self, model, optimizer):
        if self._model_state is None or self._optimizer_state is None:
            raise RuntimeError("No snapshot available to restore.")
        model.load_state_dict(self._model_state)
        optimizer.load_state_dict(self._optimizer_state)
