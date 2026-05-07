# Abstract base class for rollback controllers.
# Defines the interface: initialize(), decide() -> (accepted, info_dict).
# All controllers (no-controller, Or-epsilon, CUSUM-epsilon) inherit from this.

from abc import ABC, abstractmethod


class BaseController(ABC):
    @abstractmethod
    def initialize(self, y0):
        pass

    @abstractmethod
    def decide(self, y_proposed):
        pass
