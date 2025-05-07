import torch as th
import numpy as np
from typing import (
    TypeAlias,
    Union,
    List,
    Tuple
)
from torch.nn import functional as F
from torch.nn import (
    Conv2d,
    Linear,
    Dropout,
    Module,
    Sequential,
    ReLU,
    Softmax,
    Sigmoid,
    Parameter,
    BatchNorm2d,
    LayerNorm,
    ModuleList,
    Dropout2d,
    ConvTranspose2d,
    Tanh
)


_activations_ = {
    "relu": ReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid
}


class GlobalAVGPooling2d(Module):

    def __init__(self) -> None:
        super().__init__()
        self._activation_ = _activations_["softmax"](dim=1)

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        height_pool = inputs.mean(dim=-1)
        width_pool = height_pool.mean(dim=-1)
        return self._activation_(width_pool)

class STNet(Module):

    def __init__(
        self, 
        grid_size: Tuple[int, int],
        camera_matrix: Union[list, th.Tensor], 
        device: str="cpu",
    ) -> None:

        super().__init__()
        self.w, self.h = grid_size
        self.device = device
        self.K = camera_matrix
        if isinstance(camera_matrix, list):
            self.K = th.Tensor(camera_matrix)

        self.K_inv = th.linalg.inv(self.K)

        self._grid_ = th.stack(th.meshgrid(
            th.arange(self.h),
            th.arange(self.w),
            indexing="xy"
        ), dim=-1)

    
    def __call__(
        self,
        depth_map: th.Tensor,
        transforms: th.Tensor,
        image: th.Tensor
    ) -> th.Tensor:
        
        homo_uv = th.cat([
            self._grid_,
            th.ones((
                self._grid_.size()[0],
                self._grid_.size()[1],
                1
            ))
        ], dim=-1).unsqueeze(dim=0)
        homo_uv = homo_uv.repeat(depth_map.size()[0], 1, 1, 1)

        ppoint_inv =  depth_map * (homo_uv @ self.K_inv.view(1, 3, 3))
        homo_3d = th.cat([
            ppoint_inv,
            th.ones((
                ppoint_inv.size()[0],
                self._grid_.size()[0],
                self._grid_.size()[1],
                1
            ))
        ], dim=-1)
        
        tr_points = (homo_3d[..., :] @ transforms.view(
            transforms.size()[0],
            1, 4, 4
        ))[..., :-1]
        ppoint_dir = tr_points[..., :] @ self.K.view(1, 3, 3) 
        uv_grid = ppoint_dir[..., :-1] / (ppoint_dir[..., -1].unsqueeze(dim=-1) + 1e-4)
        uv_grid = ((2 * (uv_grid / uv_grid.max())) - 1)

        if image is not None:
            return F.grid_sample(
                image,
                grid=uv_grid,
                mode="bilinear",
            )

        else:
            return uv_grid

        

import matplotlib.pyplot as plt
plt.style.use("dark_background")
from torchvision.io import read_image
from torchvision.transforms import Resize

if __name__ == "__main__":


    res = Resize((128, 128))
    path = "C:\\Users\\1\\Desktop\\PythonProjects\\SlamVODeepML\\test\\JellFish.jpeg"
    image = read_image(path)
    image = res(image)
    image = (image / 255.0).to(th.float32).unsqueeze(dim=0)

    grid_size = (128, 128)
    camera_matrix = th.Tensor([
        [100,   0, 63.5],
        [  0, 100, 63.5],
        [  0,   0,    1]
    ])
    layer = STNet(
        grid_size=grid_size,
        camera_matrix=camera_matrix
    )
    pose_layer = PoseNet(
        in_channels=3,
        hiden_channels=32,
    )
    T = pose_layer(image)
    print(T[0])
    depth = th.normal(0, 1, (1, 128, 128, 1))
    
    warped_image = layer(
        depth_map=depth, 
        transforms=T,
        image=image
    ).squeeze(dim=0).permute(1, 2, 0)
    print(th.isnan(warped_image).any())
    

    _, axis = plt.subplots(ncols=2)
    axis[0].imshow(image.squeeze(dim=0).permute(1, 2, 0))
    axis[1].imshow(warped_image.detach())
    plt.show()
    print(warped_image.size())