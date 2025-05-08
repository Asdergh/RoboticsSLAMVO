import torch as th
import os
import json, yaml
import matplotlib.pyplot as plt 
plt.style.use("dark_background")


from tqdm import tqdm
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
from utils.functions import *


_read_io_ = {
    "json": json.load,
    "yaml": yaml.load
}

_losses_ = {
    "mse": MSELoss(),
    "cre": CrossEntropyLoss(),
    "l1": L1Loss(),
    "huber": HuberLoss()
}

_optims_ = {
    "adam": Adam
}

class SlamVOPipeLine:


    def __init__(self, loader: DataLoader, config: Union[str, dict]) -> None:
        
        self._loader_ = loader
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
        

        opt_conf = self.config["optim"]
        learning_rate = opt_conf["learning_rate"]
        type = opt_conf["type"]
        self._opt_ = _optims_[type](
            lr=learning_rate, 
            params=self.model.parameters()
        )

    
    def _save_plot_(self, path: str, values: Tuple, desc: Tuple) -> None:

        print("saving")
        fig, axis = plt.subplots(ncols=2)
        
        axis[0].imshow(values[0])
        axis[0].set_title(desc[0], rotation=5)
        axis[1].imshow(values[1])
        axis[1].set_title(desc[1], rotation=5)

        fig.savefig(path)

    def _save_res_perstep_(self, step: int) -> None:

        conf = self.config["save_res"]
        root_path = os.path.join(self.config["log_path"], f"traning_step_{step}")
        if not os.path.exists(root_path):
            os.mkdir(root_path)
        
        idx = th.randint(0, 32, (1, ))
        data_samples = next(iter(self._loader_))
        prev_frame = data_samples[1][idx]
        next_frame = data_samples[2][idx]

        with th.no_grad():
            model_out = self.model(
                (prev_frame, next_frame),
                depth_level=2
            )
        
        if conf["save_unc"]:
            unc_path = os.path.join(root_path, "uncertanty_maps.png")
            self._save_plot_(
                path=unc_path,
                values=(
                    model_out.unc_prev.squeeze(dim=0).permute(1, 2, 0),
                    model_out.unc_next.squeeze(dim=0).permute(1, 2, 0)
                ),
                desc=("Unc prev", "Unc next")
            )

        if conf["save_depth"]:
            depth_path = os.path.join(root_path, "depth_maps.png")
            self._save_plot_(
                path=depth_path,
                values=(
                    model_out.depth_prev.squeeze(dim=0).permute(1, 2, 0), 
                    model_out.depth_next.squeeze(dim=0).permute(1, 2, 0)
                ),
                desc=("Depth prev", "Depth next")
            )

        if conf["save_warping"]:
            warping_path = os.path.join(root_path, "warped_images.png")
            self._save_plot_(
                path=warping_path,
                values=(
                    model_out.image_tprev.squeeze(dim=0).permute(1, 2, 0), 
                    model_out.image_tnext.squeeze(dim=0).permute(1, 2, 0)
                ),
                desc=("Warping prev", "Warping next")
            )


    def fit(self) -> None:
        
        conf = self.config["hyper_params"]
        if conf["plot_losses"]:
            rep_losses = []
            ssim_losses = []
            unc_levels = []

        for step in range(conf["n_steps"]):

            if conf["plot_losses"]:
                rep_local = 0.0
                ssim_local = 0.0
                unc_local = 0.0
            
            if "save_res" in self.config:
                if (step % conf["steps_per_save"]) == 0:
                    print("True")
                    self._save_res_perstep_(step=step)

            for (keyframe, prev_frame, next_frame) in tqdm(
                self._loader_,
                colour="green",
                ascii="->=",
                desc="Fitting Data..."
            ):
                
                self._opt_.zero_grad()

                warping_prev, warping_next, unc_prev, unc_next, _, _ = self.model(
                    (prev_frame, next_frame),
                    depth_level=2
                )
                # experiment: try to use implement uncertanty maps right into images to see what will
                # happen with resulting nfr loss, and after try to use .mean() values from uncertanty map
                # and regularize loss objective with this values
                warping_prev = (warping_prev / unc_prev) + th.log(unc_prev)
                warping_next = (warping_next / unc_next) + th.log(unc_next)

                prev_ssim = ssim(
                    I1=keyframe,
                    I2=warping_prev,
                    kernel_size=(3, 3)
                )
                prev_l1 = _losses_["l1"](keyframe, warping_prev)
                prev_rep_loss = (conf["alpha"] / 2) * prev_ssim + (1 - conf["alpha"] * prev_l1)

                next_ssim = ssim(
                    I1=keyframe,
                    I2=warping_next,
                    kernel_size=(3, 3)
                )
                next_l1 = _losses_["l1"](keyframe, warping_next)
                next_rep_loss = (conf["alpha"] / 2) * next_ssim + (1 - conf["alpha"] * next_l1)
                
                N = (keyframe.size()[1] * keyframe.size()[2])
                rep_loss = th.min(prev_rep_loss, next_rep_loss) / N
                rep_loss.backward()
                if conf["plot_losses"]:

                    rep_local += rep_loss.item()
                    ssim_local += th.min(next_ssim, prev_ssim).item()
                    unc_local += th.min(th.mean(unc_next), th.mean(unc_prev)).item()
                    
                self._opt_.step()

            if conf["plot_losses"]:
                
                rep_losses.append(rep_local)
                ssim_losses.append(ssim_local)
                unc_levels.append(unc_levels)
        
        
        weights_path = os.path.join(self.config["log_path"], "weights.pt")
        th.save(self.model.state_dict(), weights_path)

        if conf["plot_losses"]:

            _, axis = plt.subplots(nrows=3)
            axis[0].plot(rep_losses, color="green")
            axis[1].plot(ssim_losses, color="orange")
            axis[2].plot(unc_levels, color="red")

            axis[0].set_title("reprojection loss")
            axis[1].set_title("ssim loss")
            axis[2].set_title("unc loss")

            plt.show()
            


if __name__ == "__main__":

    from utils.loaders import SLAMSterioKITILoader
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
        "log_path": "C:\\Users\\1\\Desktop\\PythonProjects\\SlamVODeepML\\test\\meta\\SLAMVoModelTraningLogs",
        "hyper_params": {
            "alpha": 8.98e-4,
            "n_steps": 1,
            "plot_losses": True,
            "steps_per_save": 5
        },
        "save_res": {
            "step_per_save": 5,
            "save_depth": True,
            "save_unc": True,
            "save_warping": True
        },
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

    path = "C:\\Users\\1\\Downloads\\rgbd_dataset_freiburg2_pioneer_slam\\rgbd_dataset_freiburg2_pioneer_slam"
    dataset = SLAMSterioKITILoader(path=path, image_size=(128, 128))
    loader = DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False
    )
    slam_vo = SlamVOPipeLine(
        config=general_conf,
        loader=loader
    ) 
    slam_vo.fit()
     
         