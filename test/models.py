import torch as th
import numpy as np
import json as js

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
    Tanh,
    ModuleDict
)

from layers import *
from utils.functions import *
from collections import namedtuple
from typing import NamedTuple

_activations_ = {
    "relu": ReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "tanh": Tanh
}


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
        device: str="cpu",
        twist_out: bool=False
    ) -> None:

        super().__init__()
        self.twist = twist_out
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
        twist = twist / th.max(twist)
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
        
        # print(T.max(), T.min())
        return T
    


class Unet(Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dp_rate: float=None,
        pyramid_depth: int=3,
        hiden_channels: int=32,
        hiden_activation: str="relu",
        out_activation: str="relu"
    ) -> None:
        
        super().__init__()
        self.pyramid_depth = pyramid_depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dp_rate = dp_rate
        self.h_activation = hiden_activation
        self.h_channels = hiden_channels
        self.out_activation = out_activation

        
        
        self._down_ = self._generate_pyramid_stack_(
            depth=pyramid_depth,
            mode="down",
            temperature=1
        )
        self._up_, self._out_heads_ = self._generate_pyramid_stack_(
            depth=pyramid_depth,
            mode="up",
            temperature=2
        )
    
    def _generate_pyramid_stack_(
        self, 
        depth: int,
        mode: str="down", 
        temperature: int=1,
    ) -> ModuleList:
        
        stack = []
        if mode == "up":
            out_heads = []

        for idx in range(depth):
            
            channels = temperature * self.h_channels
            if idx == 0:
                if mode == "down":
                    channels = self.in_channels
                else:
                    channels = self.h_channels

            if mode == "up":
                conv = ConvTranspose2d(
                    in_channels=channels,
                    out_channels=self.h_channels,
                    kernel_size=(2, 2),
                    padding=0,
                    stride=2
                )
            else:
                conv = Conv2d(
                    in_channels=channels,
                    out_channels=self.h_channels,
                    kernel_size=(3, 3),
                    padding=1,
                    stride=2
                )

            layer = Sequential( 
                conv,
                BatchNorm2d(num_features=self.h_channels),
                Dropout2d(p=self.dp_rate),
                _activations_[self.h_activation]()
            )
            stack.append(layer)

            if mode == "up":
                
                out_ch = (temperature * self.h_channels)
                if idx == (depth - 1):
                    out_ch = self.h_channels

                out_head = Sequential(
                    Conv2d(
                        in_channels=out_ch,
                        out_channels=self.out_channels,
                        kernel_size=(3, 3),
                        padding=1,
                        stride=1
                    ),
                    _activations_[self.out_activation]()
                )
                out_heads.append(out_head)
            

        if mode == "down":
            return ModuleList(stack)

        elif mode == "up":
            return (ModuleList(stack), ModuleList(out_heads))
    

    def __call__(self, inputs: th.Tensor, out_depth: int=None) -> Union[Tuple[th.Tensor], th.Tensor]:
        
        if out_depth is not None:
            assert (out_depth >= 0 and out_depth <= self.pyramid_depth - 1), """
            Outpur depth must be in range <-- [0, pyramid_depth - 1] -->. 
            Where pyramid_depth - the number of conv layers that pad dims 
            of input images !
            """

        down = []
        x = inputs
        for idx, layer in enumerate(self._down_):
            x = layer(x)
            down.append(x)
        
        down = down[::-1][1:]
        outs = []
        for idx, (up_layer, out_head) in enumerate(zip(self._up_, self._out_heads_)):

            if idx != len(self._up_) - 1:

                x = up_layer(x)
                down_out = down[idx]
                x = th.cat([x, down_out], dim=1)

                out = out_head(x)
                outs.append(out)
            
            else:
                x = up_layer(x)
                out = out_head(x)
                outs.append(out)

        

        if out_depth is not None:
            return outs[out_depth]

        else:
            return tuple(outs)
                

class DepthNet(Unet):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pyramid_depth: int=3,
        hiden_channels: int=32,
        hiden_activation: str="relu",
        out_activation: str="relu",
        depth_activation: str="relu",
        unc_actuvation: str="relu",
        dp_unet: float=0.0,
        dp_depth: float=0.0,
        dp_unc: float=0.0
    ) -> None:
        
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            dp_rate=dp_unet,
            pyramid_depth=pyramid_depth,
            hiden_activation=hiden_activation,
            hiden_channels=hiden_channels,
            out_activation=out_activation
        )
        self.out_ch = out_channels
        self.dp_depth = dp_depth
        self.dp_unc = dp_unc
        self.depth = pyramid_depth
        self.depth_act = depth_activation
        self.unc_act = unc_actuvation

        self._net_ = self._get_out_heads_(depth=pyramid_depth)
        

    def _get_out_heads_(self, depth: int) -> ModuleList:
        
        stack = []
        for _ in range(depth):
            
            params = {
                "in_channels": self.out_ch,
                "out_channels": 1,
                "kernel_size": (3, 3),
                "stride": 1,
                "padding": 1
            }
            pare = ModuleDict({
                "depth": Sequential(
                    Conv2d(**params),
                    BatchNorm2d(num_features=1),
                    Dropout2d(p=self.dp_depth),
                    _activations_[self.depth_act]()
                ),
                "unc": Sequential(
                    Conv2d(**params),
                    BatchNorm2d(num_features=1),
                    Dropout2d(p=self.dp_unc),
                    _activations_[self.unc_act]()
                )
            })
            stack.append(pare)
        
        stack = ModuleList(stack)
        return stack
        
        
    def __call__(self, inputs: th.Tensor, out_depth: int=None) -> th.Tensor:

        if out_depth is not None:
            
            unet_out = super().__call__(inputs, out_depth=out_depth)
            (depth_head, unc_head) = self._net_[out_depth].values()
            return (depth_head(unet_out), unc_head(unet_out))

        else:

            unet_out = super().__call__(inputs)
            outs = []
            for idx in range(self.pyramid_depth):

                (depth_head, unc_head) = self._net_[idx].values()
                depth_out = depth_head(unet_out[idx])
                unc_out = unc_head(unet_out[idx])
                outs.append((depth_out, unc_out))
            
            return tuple(outs)
        

class SlamD3VO(Module):

    def __init__(
        self,
        depthnet_conf: Union[str, dict],
        posenet_conf: Union[str, dict],
        warping_conf: Union[str, dict]
    ) -> None:
        
        super().__init__()
        self._depth_estimator_ = DepthNet(**depthnet_conf)
        self._pose_estimator_ = PoseNet(**posenet_conf)
        self._warping_layer_ = STNet(**warping_conf)

        self._predict_tuple_ = namedtuple("SlamVOPredictTuple", [
            "depth",
            "uncertanty",
            "odometry",
        ])
        self._output_tuple_ = namedtuple("SlamVOOutTuple", [
            "image_tprev",
            "image_tnext",
            "unc_prev",
            "unc_next",
            "depth_prev",
            "depth_next"
        ])

    
    def predict(self, inputs: th.Tensor):

        with th.no_grad():
            depth, unc = self._depth_estimator_(
                inputs,
                out_depth=2
            )
            T = self._pose_estimator_(inputs)
            return self._predict_tuple_(
                depth=depth,
                uncertanty=unc,
                odometry=T
            )
        

    def __call__(
        self, 
        inputs: Tuple, 
        depth_level: int=None
    ) -> NamedTuple:

        depth_prev, unc_prev = self._depth_estimator_(
            inputs[0],
            out_depth=depth_level
        )
        depth_next, unc_next = self._depth_estimator_(
            inputs[1],
            out_depth=depth_level
        )

        if depth_level is not None:
            T_prev = self._pose_estimator_(inputs[0])
            T_next = self._pose_estimator_(inputs[1])
            
            prev_warping = self._warping_layer_(
                depth_map=depth_prev.permute(0, 2, 3, 1),
                transforms=T_prev,
                image=inputs[0]
            )
            next_warping = self._warping_layer_(
                depth_map=depth_next.permute(0, 2, 3, 1),
                transforms=T_next,
                image=inputs[1]
            )

            # print(f"""
            #     prev_warping: {prev_warping.min()}, {prev_warping.max()},
            #     next_warping: {next_warping.min()}, {next_warping.max()},
            #     unc_prev: {unc_prev.min()}, {unc_prev.max()},
            #     unc_next: {unc_next.min()}, {unc_next.max()},
            #     depth_prev: {depth_prev.min()}, {depth_prev.max()},
            #     depth_next: {depth_next.min()}, {depth_next.max()}
            # """)
            return self._output_tuple_(
                image_tprev=prev_warping,
                image_tnext=next_warping,
                unc_prev=unc_prev,
                unc_next=unc_next,
                depth_prev=depth_prev,
                depth_next=depth_next
            )

            

            
        

if __name__ ==  "__main__":
    
    test = th.normal(0.0, 1.0, (10, 3, 128, 128))


    pose_config = {
        "in_channels": 3,
        "hiden_channels": 32,
        "kernel_size": (3, 3),
        "padding": 1,
        "stride": 1,
        "activation": "relu",
        "dp_rate": 0.0,
        "normalize_conv": False,
        "normalize_linear": False,
        "embedding_dim": 32,
        "device": "cpu",
        "twist_out": False
    }

    depth_config = {
        "in_channels": 3,
        "out_channels": 1,
        "pyramid_depth": 3,
        "hiden_activation": "relu",
        "hiden_channels": 32,
        "out_activation": "relu",
        "dp_unet": 0.45,
        "dp_depth": 0.45,
        "dp_unc": 0.34,
        "depth_activation": "sigmoid",
        "unc_actuvation": "softmax" 
    }
    waripng_conf = {
        "camera_matrix": [
            [100,   0, 63.5],
            [  0, 100, 63.5],
            [  0,   0,    1]
        ],
        "grid_size": (128, 128)
    }

    slam_vo_model = SlamD3VO(
        depthnet_conf=depth_config,
        posenet_conf=pose_config,
        warping_conf=waripng_conf
    ) 

    test_img = th.normal(0, 1, (1, 3, 128, 128))
    inputs_images = namedtuple("ImageInput", [
        "key",
        "left",
        "right"
    ])
    images = inputs_images(
        key=th.normal(0, 1, (1, 3, 128, 128)),
        left=th.normal(0, 1, (1, 3, 128, 128)),
        right=th.normal(0, 1, (1, 3, 128, 128))
    )
    out = slam_vo_model(images, depth_level=2)
    print(out.image_tprev.size(), out.image_tnext.size())
    
   
    
    
    