import math
import numbers
from typing import Optional, Union, List, Tuple

import torch
from torchvision import transforms
from torchvision.transforms.functional import (
    InterpolationMode, get_dimensions, resize, crop, resized_crop, center_crop
)


class MinimalistRandomResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        num_epochs,
        scale=(0.08, 1.0),
        sched_func="convex",
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__(size, scale, interpolation=interpolation, antialias=antialias)
        self.num_epochs = num_epochs
        self.sched_func = sched_func
        self.iter = 0

    # Call step() after each epoch.
    def step(self):
        self.iter += 1

    def get_min_scale(self) -> float:
        t = self.iter / (self.num_epochs - 1)
        assert 0 <= t <= 1

        if self.sched_func == "min":
            s = 0
        elif self.sched_func == "max":
            s = 1
        elif self.sched_func == "linear":
            s = t
        elif self.sched_func == "convex":
            s = 1 - math.sqrt(1 - t ** 2)
        elif self.sched_func == "concave":
            s = math.sqrt(1 - (1 - t) ** 2)
        else:
            raise ValueError
        assert 0 <= s <= 1

        return s * (self.scale[1] - self.scale[0]) + self.scale[0]

    @staticmethod
    def get_params(img, scale: List[float]) -> Tuple[int, int, int, int]:
        """Get parameters for ``crop`` for a random sized crop.

        Args:
            img (PIL Image or Tensor): Input image.
            scale (list): range of scale of the origin size cropped

        Returns:
            tuple: params (i, j, h, w) to be passed to ``crop`` for a random
            sized crop.
        """
        _, height, width = get_dimensions(img)
        area = height * width

        min_scale = min(scale[0], min(height, width) ** 2 / area)
        max_scale = min(scale[1], min(height, width) ** 2 / area)
        target_area = area * torch.empty(1).uniform_(min_scale, max_scale).item()

        h = int(round(math.sqrt(target_area)))
        w = int(round(math.sqrt(target_area)))
        i = torch.randint(0, height - h + 1, size=(1,)).item()
        j = torch.randint(0, width - w + 1, size=(1,)).item()

        return i, j, h, w

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        min_scale = self.get_min_scale()
        max_scale = self.scale[1]
        i, j, h, w = self.get_params(img, [min_scale, max_scale])
        return resized_crop(img, i, j, h, w, self.size, self.interpolation, antialias=self.antialias)


def non_overlapping_five_crop(img, size, reverse=False):
    """ modified from torchvision.transforms.functional.five_crop
    """
    if isinstance(size, numbers.Number):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    _, image_height, image_width = get_dimensions(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        msg = "Requested crop size {} is bigger than input size {}"
        raise ValueError(msg.format(size, (image_height, image_width)))

    def _fractional_crop(top: float, left: float):
        crop_top = int(round((image_height - crop_height) * top))
        crop_left = int(round((image_width - crop_width) * left))
        return crop(img, crop_top, crop_left, crop_height, crop_width)

    if not reverse:
        top_left_list = [(0.25, 0.0), (0.0, 0.75), (1.0, 0.25), (0.75, 1.0), (0.5, 0.5)]
    else:
        top_left_list = [(0.0, 0.25), (0.25, 1.0), (0.75, 0.0), (1.0, 0.75), (0.5, 0.5)]
    
    return tuple(_fractional_crop(top, left) for top, left in top_left_list)


class NonOverlappingFiveCrop(transforms.FiveCrop):
    def __init__(self, size, reverse=False):
        super().__init__(size)
        self.reverse = reverse
    
    def forward(self, img):
        return non_overlapping_five_crop(img, self.size, self.reverse)


class MultiScaleResizedCrop(transforms.RandomResizedCrop):
    def __init__(
        self,
        size,
        scale=(0.5, 1.0),
        steps: int = 5,
        interpolation=InterpolationMode.BILINEAR,
        antialias: Optional[bool] = True,
    ):
        super().__init__(size, scale, interpolation=interpolation, antialias=antialias)
        self.steps = steps

    @staticmethod
    def resized_center_crop(img, crop_size, output_size, interpolation, antialias) -> Tuple[int, int, int, int]:
        img = center_crop(img, crop_size)
        img = resize(img, output_size, interpolation, antialias=antialias)
        return img
    
    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be cropped and resized.

        Returns:
            PIL Image or Tensor: Randomly cropped and resized image.
        """
        _, height, width = get_dimensions(img)
        center_area = min(height, width) ** 2

        imgs = []
        for scale in torch.linspace(self.scale[0], self.scale[1], self.steps):
            target_area = center_area * scale
            h = int(round(math.sqrt(target_area)))
            w = int(round(math.sqrt(target_area)))
            imgs.append(self.resized_center_crop(img, [h, w], self.size, interpolation=self.interpolation, antialias=self.antialias))
        return tuple(imgs)