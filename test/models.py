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
from utils import *

_activations_ = {
    "relu": ReLU,
    "softmax": Softmax,
    "sigmoid": Sigmoid,
    "tanh": Tanh
}

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
    

if __name__ ==  "__main__":
    
    test = th.normal(0.0, 1.0, (10, 3, 128, 128))
    depthnet = DepthNet(
        in_channels=3,
        out_channels=1,
        pyramid_depth=3,
        hiden_activation="relu",
        hiden_channels=32,
        out_activation="relu",
        dp_unet=0.45,
        dp_depth=0.45,
        dp_unc=0.34,
        depth_activation="sigmoid",
        unc_actuvation="sigmoid" 
    )
    
    outs = depthnet(test, out_depth=3)
    print(outs[0].size(), outs[1].size())
    # for out in outs:
    #     print(out[0].size(), out[1].size())

    # unet = Unet(
    #     in_channels=3,
    #     out_channels=1,
    #     dp_rate=0.45,
    #     pyramid_depth=3,
    #     hiden_activation="relu",
    #     hiden_channels=32,
    #     out_activation="relu"
    # )
    
    # outs = unet(test)
    # for out in outs:
    #     print(out.size())
    
    
    