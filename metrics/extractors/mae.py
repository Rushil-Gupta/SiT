import torch
import torch.nn as nn
from torchvision import transforms as T
from ..base import FeatureExtractor
from ..registry import register
from ..utils import ImageClamper
from dataset.ops_dataset import GlobalMinMaxNorm

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

@register("openphenom")
class MAEMinmaxExtractor(FeatureExtractor):
    def _load_model(self):
        return _load_mae_model(self.device)

    def _preprocess(self, images, gen=False):
        # images = torch.nn.functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
        if gen:
            transforms_pipeline = T.v2.Compose([
                ImageClamper(minv=-1.0, maxv=1.0),
                T.v2.Resize((256, 256)),
                ImageClamper(minv=0.0, maxv=1.0),
            ])
        else:
            transforms_pipeline = T.v2.Compose([
                T.v2.Resize((256, 256)),
                GlobalMinMaxNorm(),
            ])
        images = transforms_pipeline(images)
        return images

    def _forward(self, images):
        return self.model.predict(images)

    @property
    def dim(self):
        return 384

    @property
    def name(self):
        # return "mae_minmax"
        return "openphenom"


# @register("mae_arcsinh")
# class MAEArcsinhExtractor(FeatureExtractor):
#     def _load_model(self):
#         return _load_mae_model(self.device)

#     def _preprocess(self, images):
#         images = torch.nn.functional.interpolate(images, size=(256, 256), mode="bilinear", align_corners=False)
#         images = invert_minmax(images)
#         images = torch.arcsinh(images)
#         images = (images - 7.0) / 7.0
#         return images

#     def _forward(self, images):
#         return self.model.predict(images)

#     @property
#     def dim(self):
#         return 384

#     @property
#     def name(self):
#         return "mae_arcsinh"
