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


class PoseNet(Module):

    def __init__(
        self,
        in_channels: int,
        hiden_channels: int,
        kernel_size: Union[Tuple[int, int], int]=(3, 3),
        padding: int=1,
        stride: Union[Tuple[int, int], int]=1,
        activation: str = "relu",
        dp_rate: float = 0.0,
        normalize_conv: bool = False,
        normalize_linear: bool = False,
        embedding_dim: int = 32,
        device: str="cpu"
    ) -> None:

        super().__init__()
        self.embedding_dim = embedding_dim
        self.device = device
        
        self._conv_ = [
            Conv2d(
                in_channels=in_channels,
                out_channels=hiden_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride
            ),
            _activations_[activation]()
        ]
        if normalize_conv:
            self._conv_ += [BatchNorm2d(num_features=hiden_channels)]
        
        self._conv_ = ModuleList(self._conv_)
        for layer in self._conv_:
            if isinstance(layer, (Linear, Conv2d)):
                th.nn.init.normal_(layer.weight, mean=0.0, std=1.0)

        self._linear_ = [
            Linear(self.embedding_dim, 6),
            Dropout(p=dp_rate)
        ]
        if normalize_linear:
            self._linear_ += [LayerNorm(num_features=6)]
        
        self._linear_ = Sequential(*self._linear_)
    

    def __call__(self, inputs: th.Tensor) -> th.Tensor:

        x = inputs
        for layer in self._conv_:
            x = layer(x)

        x = th.flatten(x, start_dim=1, end_dim=-1)
        embedding_weights = th.normal(0.0, 1.0, (x.size()[1], self.embedding_dim)).to(self.device)
        embedding = F.linear(x, weight=embedding_weights.T)
        
        twist = self._linear_(embedding)
        gamma = twist[:, :3]
        t = twist[:, 3:]
        
        theta = th.norm(gamma, keepdim=True, dim=-1)
        gamma = gamma / (theta + 1e-5)
        theta = theta.view(theta.size()[0], 1, 1)

        cross_matrix = th.zeros((inputs.size()[0], 3, 3)).to(self.device)
        cross_matrix[..., 0, 1] = -gamma[:, -1]
        cross_matrix[..., 0, 2] = gamma[:, 1]
        cross_matrix[..., 1, 0] = gamma[:, -1]
        cross_matrix[..., 1, -1] = -gamma[:, 0]
        cross_matrix[..., 2, 0] = -gamma[:, 1]
        cross_matrix[..., 2, 1] = gamma[:, 0]

        rotation = th.zeros((inputs.size()[0], 3, 3)).to(self.device)
        rotation[:, :, :] = th.eye(3).to(self.device) + (th.sin(theta) * cross_matrix) + ((1 - th.cos(theta)) * (cross_matrix @ cross_matrix))

        radriges_rotation = th.zeros((inputs.size()[0], 3, 3)).to(self.device)
        radriges_rotation = th.eye(3).to(self.device) + (((1 - th.cos(theta)) / ((theta ** 2) + 1e-5)) * cross_matrix) + (((theta - th.sin(theta)) / ((theta ** 3) + 1e-5)) * (cross_matrix @ cross_matrix))


        t_rotated = t.view(t.size()[0], 1, 3) @ radriges_rotation
        t_rotated = t_rotated.permute(0, 2, 1)
        
        T = th.zeros((inputs.size()[0], 4, 4)).to(self.device)

        T[:, :3, :3] = rotation
        T[:, :3, 3:] = t_rotated
        
        return T

class STNet(Module):

    def __init__(
        self, 
        grid_size: Tuple[int, int], 
        device: str="cpu"
    ) -> None:

        super().__init__()
        self.grid_size = grid_size
        self.device = device
    
    def __call__(
        self, 
        img: th.Tensor, 
        kamera_matrix: th.Tensor, 
        transforms: th.Tensor,
        depth_map: th.Tensor
    ) -> th.Tensor:
        
        tmp_ones = th.ones((img.size[0], img.size[2], img.size[3], 1)).to(self.device)
        if len(depth_map) < 4:
            depth_map = depth_map.unsqueeze(dim=-1)
            
        u, v = th.meshgrid(
            th.arange(img.size()[2]),
            th.arange(img.size()[3])
        ).to(self.device)

        pix = th.stack([u, v], dim=1).repeat(img.size()[0], 1, 1, 1)
        pix = th.cat([pix, tmp_ones], dim=-1)
        depthed_pix = depth_map * pix

        pix_3dK = depthed_pix[...] @ th.linalg.inv(kamera_matrix)
        pix_3dK = th.cat([pix, tmp_ones], dim=-1)
        
        pix_3dTranspose = transforms @ pix_3dK
        pix_3dTranspose = pix_3dTranspose[..., :-1]
        pix_3dTranspose = pix_3dTranspose @ kamera_matrix

        pix_3d2d = pix_3dTranspose[..., :-1] / pix_3dTranspose[..., -1]
        
        
        
        


        


        
        

        

        
        




if __name__ == "__main__":

    test = th.normal(0.0, 1.1, (10, 3, 128, 128))
    stn = PoseNet(
        in_channels=3,
        hiden_channels=3,
        dp_rate=0.45
    )
    out = stn(test).mean()
    out.backward()
    # linear = Linear(3, 128)
    # GAP = GlobalAVGPooling2d()
    
    # out = linear(GAP(test)).mean()
    # print(out.size())
    # out.backward()
    # print(out.grad_fn)
