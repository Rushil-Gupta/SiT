import torch
from ..base import FeatureExtractor
from ..registry import register
from ..utils import to_rgb, IMAGENET_MEAN, IMAGENET_STD


@register("dinov2")
class DINOv2Extractor(FeatureExtractor):
    def _load_model(self):
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14", verbose=False)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _preprocess(self, images):
        images = to_rgb(images)
        images = torch.nn.functional.interpolate(images, size=(224, 224), mode="bilinear", align_corners=False)
        mean = IMAGENET_MEAN[None, :, None, None].to(images.device)
        std = IMAGENET_STD[None, :, None, None].to(images.device)
        return (images - mean) / std

    def _forward(self, images):
        tokens = self.model.forward_features(images)
        return tokens["x_norm_patchtokens"].mean(dim=1)

    @property
    def dim(self):
        return 768

    @property
    def name(self):
        return "dinov2"
