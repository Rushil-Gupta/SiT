import torch
import torch.nn as nn
from torchvision import transforms as T
from ..base import FeatureExtractor
from ..registry import register
from ..utils import invert_minmax


def _load_mae_model(device):
    from OpenPhenom.huggingface_mae import MAEModel
    model = MAEModel.from_pretrained(
        "recursionpharma/OpenPhenom",
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    model.input_norm = nn.Identity()
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


@register("mae_minmax")
class MAEMinmaxExtractor(FeatureExtractor):
    def _load_model(self):
        return _load_mae_model(self.device)

    def _preprocess(self, images):
        images = torch.nn.functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        return images

    def _forward(self, images):
        return self.model.predict(images)

    @property
    def dim(self):
        return 384

    @property
    def name(self):
        return "mae_minmax"


@register("mae_arcsinh")
class MAEArcsinhExtractor(FeatureExtractor):
    def _load_model(self):
        return _load_mae_model(self.device)

    def _preprocess(self, images):
        images = torch.nn.functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        images = invert_minmax(images)
        images = torch.arcsinh(images)
        images = (images - 7.0) / 7.0
        return images

    def _forward(self, images):
        return self.model.predict(images)

    @property
    def dim(self):
        return 384

    @property
    def name(self):
        return "mae_arcsinh"
