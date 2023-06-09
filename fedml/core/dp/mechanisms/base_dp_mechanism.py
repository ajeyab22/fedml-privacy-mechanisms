from abc import ABC, abstractmethod


class BaseDPMechanism(ABC):
    @abstractmethod
    def compute_noise(self, size):
        pass
