import torch
from torchvision.transforms import v2
from ..base import FeatureExtractor
from ..registry import register
from ..utils import ConvtoRGB, IMAGENET_MEAN, IMAGENET_STD, ImageClamper
from dataset.ops_dataset import GlobalMinMaxNorm


@register("cell_dino")
class CellDINOExtractor(FeatureExtractor):
    def _load_model(self):
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _preprocess(self, images, gen=False):
        # DINOv2 ViT-L/14's patch embedding is Conv2d(3, ...), so the 4-channel
        # microscopy input must be pseudo-colored to RGB first (same as inception.py).
        if gen:
            transforms_pipeline = v2.Compose([
                v2.Resize((224, 224)),
                ImageClamper(minv=0.0, maxv=1.0),
                ConvtoRGB(in_channels=images.shape[1], out_channels=3),
            ])
        else:
            transforms_pipeline = v2.Compose([
                v2.Resize((224, 224)),
                GlobalMinMaxNorm(),
                ConvtoRGB(in_channels=images.shape[1], out_channels=3),
            ])
        images = transforms_pipeline(images)
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
        return "cell_dino"
