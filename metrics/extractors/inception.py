import torch
import torchvision.transforms as T
import torchvision.models as models
from ..base import FeatureExtractor
from ..registry import register
from ..utils import ConvtoRGB, IMAGENET_MEAN, IMAGENET_STD, ImageClamper
from dataset.ops_dataset import GlobalMinMaxNorm


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

    def _preprocess(self, images, gen=False):
        if gen:
            transforms_pipeline = T.v2.Compose([
                T.v2.Resize((299, 299)),
                ImageClamper(minv=0.0, maxv=1.0),
                ConvtoRGB(in_channels=images.shape[1], out_channels=3),
            ])
        else:
            transforms_pipeline = T.v2.Compose([
                T.v2.Resize((299, 299)),
                GlobalMinMaxNorm(),
                ConvtoRGB(in_channels=images.shape[1], out_channels=3),
            ])
        images = transforms_pipeline(images)
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
