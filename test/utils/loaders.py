import os
import json, yaml
import torchvision as thv
import torch as th
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image
from torchvision.transforms import Resize
from typing import Union, Tuple, NamedTuple
from collections import namedtuple






class SLAMSterioKITILoader(Dataset):

    def __init__(
        self, 
        image_size: Tuple[int], 
        path: str,
        samples_n: int=100,
        out_tensors: str="pt",
    ) -> None:

        super().__init__()

        self._samples_n_ = samples_n
        self._res_ = Resize(image_size)
        self.w, self.h = image_size
        self.out_tensors = out_tensors

        self._out_tuple_ = namedtuple("OutTuple", [
            "image_rgb",
            "depth",
            "rotation",
            "translation",
            "w", "h"
        ])

        self._depth_root_ = os.path.join(path, "depth")
        self._rgb_root_ = os.path.join(path, "rgb")
        self._depths_ = [
            os.path.join(path, "depth", file)
            for file in os.listdir(self._depth_root_)
        ]
        self._images_ = [
            os.path.join(path, "rgb", file)
            for file in os.listdir(self._rgb_root_)
        ]

        trajectory_path = os.path.join(path, "groundtruth.txt")
        trajectory = np.loadtxt(trajectory_path)
        self._translations_ = trajectory[0::15, 1:4]
        self._rotations_ = trajectory[0::15, 4:]

       
    
    def __len__(self) -> int:
        return len(self._depths_)
    

    def __getitem__(self, idx: int) -> Tuple:
        
        if idx == 0:
            idx = 1
        
        elif idx == len(self._depths_):
            idx = len(self._depths_)
        

        image_key = (self._res_(read_image(self._images_[idx])) / 255.0)
        image_prev = (self._res_(read_image(self._images_[idx - 1])) / 255.0)
        image_next = (self._res_(read_image(self._images_[idx + 1])) / 255.0)
        
        return (image_key, image_prev, image_next)


    def __iter__(self):

        for idx, (depth, image, translation, rotation) in enumerate(zip(
            self._depths_,
            self._images_,
            self._translations_,
            self._rotations_
        )):
            
            if idx == self._samples_n_:
                break

            image = (self._res_(read_image(image)).permute(1, 2, 0) / 255.0).numpy()
            depth = (self._res_(read_image(depth)).permute(1, 2, 0) / 255.0).numpy()
            if self.out_tensors == "np":
                image = image.numpy()
                depth = depth.numpy()

            yield self._out_tuple_(
                image_rgb=image,
                depth=depth,
                rotation=rotation,
                translation=translation,
                w=self.w,
                h=self.h
            )
            self._prev_rotation_ = rotation
            self._prev_trnaslation_ = translation
    




if __name__ == "__main__":
    from tqdm import tqdm
    path = "C:\\Users\\1\\Downloads\\rgbd_dataset_freiburg2_pioneer_slam\\rgbd_dataset_freiburg2_pioneer_slam"
    loader = SLAMSterioKITILoader(path=path, image_size=(1200, 1920))
    # path = "C:\\Users\\1\\Downloads\\rgbd_dataset_freiburg2_pioneer_slam\\rgbd_dataset_freiburg2_pioneer_slam"

    loader = DataLoader(
        dataset=loader,
        batch_size=32,
        shuffle=False
    )
    for sample in tqdm(loader):
        pass