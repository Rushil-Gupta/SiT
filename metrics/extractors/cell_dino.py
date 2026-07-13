import torch
from ..base import FeatureExtractor
from ..registry import register


@register("cell_dino")
class CellDINOExtractor(FeatureExtractor):
    def _load_model(self):
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", verbose=False)
        model = model.to(self.device)
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _preprocess(self, images):
        # TODO: DINOv2 ViT-L/14 patch embedding is Conv2d(3, ...) — will crash on 4ch input.
        # For now, just resize. The channel mismatch must be resolved separately.
        return torch.nn.functional.interpolate(images, size=(224, 224),
                                               mode="bilinear", align_corners=False)

    def _forward(self, images):
        tokens = self.model.forward_features(images)
        return tokens["x_norm_patchtokens"].mean(dim=1)

    @property
    def dim(self):
        return 768

    @property
    def name(self):
        return "cell_dino"
