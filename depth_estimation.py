import math
import itertools
from functools import partial
import sys
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np

import urllib
import mmcv
from mmcv.runner import load_checkpoint

import matplotlib
from torchvision import transforms

from dinov2.dinov2.eval.depth.models import build_depther

"""
Purpose of this file:
This file will handle running inference for depth estimation on an image fed from VISTA display.render() (might change input format)
It will return the depth estimation jpg or png result to later be used in downstream inputs to BarrierNet
"""

### NOTE: I will first for debugging use filepaths for image storage to make sure the pipeline works, then I will find a way to call this method
### by passing along an Image object so that it can work end-to-end as a standard function call

# this is the hardcoded image that will be used for input in debugging phase, choosing frame0.jpg, good candidate, as it has curvature shown in ado vehicle
custom_input_image = Image.open("out_videos/frame0.jpg")


def run_depth_inference(image, backbone_size, head_dataset, head_type):
    """
    image: Image type, image in which depth_inference will be run on
    backbone_size: string from set {'small', 'base', 'large', 'giant'}, will be the overall size of the DINOv2 backbone
    head_dataset: string from set {'nyu', 'kitti'}, dataset head pretrained on, recommended to use 'nyu'
    head_type: string from set {'linear', 'linear4', 'dpt'}, recommended to use 'dpt'
    ----------------------------------------------------------------------------------
    returns: Image type object that is the depth_estimate image of image input
    """
    # following documentation found at https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb 
    # loads correct backbone given backbone_size parameter
    backbone_archs = {
    "small": "vits14",
    "base": "vitb14",
    "large": "vitl14",
    "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone_size]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()
    # backbone_model.cuda()
    backbone_model.cpu() # using cpu instead of cuda GPU since we are running on MIT SuperCloud, and there are more CPUs available

    # loads pretrained head as specified by parameters
    HEAD_DATASET = head_dataset # in ("nyu", "kitti")
    HEAD_TYPE = head_type # in ("linear", "linear4", "dpt")


    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    cfg_str = load_config_from_url(head_config_url)
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    model = create_depther(
        cfg,
        backbone_model=backbone_model,
        backbone_size=backbone_size,
        head_type=HEAD_TYPE,
    )

    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.eval()
    # model.cuda()
    model.cpu() # using cpu instead of cuda GPU since we are running on MIT SuperCloud, and there are more CPUs available

    # perform inferencing
    transform = make_depth_transform()
    scale_factor = 1
    rescaled_image = image.resize((scale_factor * image.width, scale_factor * image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cpu() # Make a batch of one image, changed to cpu

    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)

    depth_image = render_depth(result.squeeze().cpu())

    # return depth_image
    return depth_image

### NOTE: EVERYTHING BELOW THIS LINE IS DIRECTLY FROM https://github.com/facebookresearch/dinov2/blob/main/notebooks/depth_estimation.ipynb 
class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])

def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)

    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True) # ((1)xhxwx4)
    colors = colors[:, :, :3] # Discard alpha component
    return Image.fromarray(colors)

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


if __name__ == "__main__":
    # for debugging purposes
    output_image = run_depth_inference(image=custom_input_image, backbone_size='base', head_dataset='nyu', head_type='dpt')
    output_image.save("depth_output/depth_frame01.jpg")
