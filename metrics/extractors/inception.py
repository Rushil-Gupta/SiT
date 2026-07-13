import torch
import torchvision.transforms as T
import torchvision.models as models
from ..base import FeatureExtractor
from ..registry import register
from ..utils import to_rgb, IMAGENET_MEAN, IMAGENET_STD


@register("inception")
class InceptionExtractor(FeatureExtractor):
    def _load_model(self):
        model = models.inception_v3(weights="DEFAULT", transform_input=False)
        model.fc = torch.nn.Identity()
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _preprocess(self, images):
        images = to_rgb(images)
        images = torch.nn.functional.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN[None, :, None, None].to(images.device)
        std = IMAGENET_STD[None, :, None, None].to(images.device)
        return (images - mean) / std

    def _forward(self, images):
        return self.model(images)

    @property
    def dim(self):
        return 2048

    @property
    def name(self):
        return "inception"
