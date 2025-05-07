import torch as th
import os
import json, yaml

from torch.utils.data import DataLoader
from torch.nn import (
    Module,
    MSELoss,
    CrossEntropyLoss,
    L1Loss,
    HuberLoss
)
from torch.optim import Optimizer, Adam

from typing import Union, List
from collections import namedtuple
from models import SlamD3VO
from test.functions import *


_read_io_ = {
    "json": json.load,
    "yaml": yaml.load
}

_losses_ = {
    "mse": MSELoss,
    "cre": CrossEntropyLoss,
    "l1": L1Loss,
    "huber": HuberLoss
}

_optims_ = {
    "adam": Adam
}

class SlamVOPipeLine:


    def __init__(self, config: Union[str, dict]) -> None:

        self.config = config
        if isinstance(config, str):
            with open(config, "r") as file:
                f_type = os.path.basename(config).split(".")[1]
                self.config = _read_io_[f_type](file)

        model_conf = self.config["model"]
        self.model = SlamD3VO(
            depthnet_conf=model_conf["depthnet_params"],
            posenet_conf=model_conf["posenet_params"],
            warping_conf=model_conf["warping_params"]
        )

        self.out_tuple = namedtuple("ConverterOutputTuple", [
            "depth",
            "uncertanty",
            "odometry_T"
        ])

        if "weigths" in self.config:
            weights = th.load(
                self.conf["weights"], 
                weights_only=True
            )
            self.model.load_state_dict(weights)
        
        self._rep_loss_ = _losses_[self.config["rep_loss"]]
        opt_conf = self.config["optim"]
        learning_rate = opt_conf["learning_rate"]
        type = opt_conf["type"]
        self._opt_ = _optims_[type](
            lr=learning_rate, 
            params=self.model.parameters()
        )

    
    def fit(self, loader: DataLoader) -> None:
        

        for _ in range(self.config["n_steps"]):
            if self.config["plot_losses"]:
                losses = []

            for (keyframe, prev_frame, next_frame) in loader:
                
                self._opt_.zero_grad()
                warping_prev, warping_next = self.model(
                    inputs=(
                        prev_frame,
                        next_frame
                    ),
                    depth_conf=2
                )

                
                
                

                
                
                

                

        




if __name__ == "__main__":


    depth_conf = {
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
    pose_conf = {
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

    warping_conf = {
        "camera_matrix": [
            [100,   0, 63.5],
            [  0, 100, 63.5],
            [  0,   0,    1]
        ],
        "grid_size": [128, 128]
    }
    general_conf = {
        "rep_loss": "mse",
        "alpha": 1e-2,
        "optim": {
            "type": "adam",
            "learning_rate": 1e-2
        },
        "model": {
            "depthnet_params": depth_conf,
            "posenet_params": pose_conf,
            "warping_params": warping_conf
        }
    }

    slam_vo = SlamVOPipeLine(general_conf)       