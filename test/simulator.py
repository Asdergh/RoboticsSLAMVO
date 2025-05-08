import numpy as np
import os
import json, yaml
import rerun as rr
import torch as th

from scipy.spatial.transform import Rotation as R
from typing import Generator, Union, Tuple, NamedTuple
from utils.loaders import SLAMSterioKITILoader
from tqdm import tqdm

class Simulator:

    def __init__(self, loader: Generator[NamedTuple, None, None], focal: float=100.0) -> None:

        self._loader_ = loader
        self._origin_ = "world"
        self.focal = focal
        self._build_simulation_()
    
    def log_camera(self, sample: NamedTuple) -> None:
        
        if hasattr(sample, "rotation"):
            quat = sample.rotation
            rotation = R.from_quat(quat).as_matrix()

        rr.log(
            f"{self._origin_}/camera",
            rr.Transform3D(
                mat3x3=rotation,
                translation=(sample.translation)
            )
        )
        rr.log(
            f"{self._origin_}/camera",
            rr.Pinhole(
                width=sample.w,
                height=sample.h,
                focal_length=self.focal
            )
        )
        rr.log(f"{self._origin_}/camera/img", rr.Image(sample.image_rgb))
        rr.log(
            f"{self._origin_}/camera/depth",
            rr.DepthImage(
                sample.depth,
                meter=67,
                colormap=5
            )
        )

    
    def _build_simulation_(self) -> None:

        rr.init("slam_vo_simulation", spawn=True)
        rr.set_time("tick", sequence=0)

        for t, sample in enumerate(tqdm(
            self._loader_,
            colour="green",
            ascii="=>-",
            desc="Rendering scene..."
        )):
            rr.set_time("tick", sequence=t + 1)
            self.log_camera(sample)


if __name__ == "__main__":

    path = "C:\\Users\\1\\Downloads\\rgbd_dataset_freiburg2_pioneer_slam\\rgbd_dataset_freiburg2_pioneer_slam"
    loader = SLAMSterioKITILoader(
        path=path, 
        image_size=(256, 256),
        samples_n=2000
    )
    sim = Simulator(loader=loader)
        