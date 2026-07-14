from abc import ABC, abstractmethod
import torch


class FeatureExtractor(ABC):
    def __init__(self, device="cuda"):
        self.device = device
        self._model = None

    @abstractmethod
    def _load_model(self):
        pass

    @property
    def model(self):
        if self._model is None:
            self._model = self._load_model()
        return self._model

    @abstractmethod
    def _preprocess(self, images):
        pass

    def encode(self, images, gen=False):
        images = self._preprocess(images, gen=gen)
        with torch.no_grad():
            features = self._forward(images)
        return features

    @abstractmethod
    def _forward(self, images):
        pass

    @property
    @abstractmethod
    def dim(self):
        pass

    @property
    @abstractmethod
    def name(self):
        pass
