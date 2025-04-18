import math
from typing import Optional, List, Tuple

import torch
from torchvision import transforms
from torchvision.transforms.functional import (
    InterpolationMode, get_dimensions, resized_crop
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
