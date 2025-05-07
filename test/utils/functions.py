import numpy as np
import torch as th
from torch.nn import Conv2d
from torch.nn import functional as F
from typing import (
    TypeAlias,
    Union,
    List,
    Tuple
)

Tensor: TypeAlias = np.ndarray | th.Tensor
Ppoints: TypeAlias = List[Union[th.Tensor, np.ndarray, tuple]]



def gauss_kernel(size: Union[Tuple[int, int], int], var: float=5e-0) -> Tensor:

    if isinstance(size, tuple):
        size = size[0]

    u = th.linspace(-size, size, size)
    g = th.exp(-(u ** 2) / (2 * var ** 2))
    g /= g.sum()

    return g.outer(g)


_kernels_ = {
    "gauss": gauss_kernel
}

def smooth_loss(
    img: Tensor, 
    depth: Tensor, 
    points: Ppoints,
    filter_type: str="sobel",
    kernel_size: Tuple[int, int]=(3, 3),
    temperature: float=1e-10,
):


    assert (filter_type in ["sobel", "laplas", "schar"]), f"""
    anknown filter type. All possibile types: 
                <-- [sobel] -->
                <-- [laplas] --> 
                <-- [schar] -->
    you passed: 
                <-- filter_type: [{filter_type}] -->
    """
    assert (type(img) == type(depth)), f"""
    Inputs must be the same type
    <-- img: [{type(img)}] -->
    <-- depth: [{type(depth)}] -->
    """

    assert (filter_type)
    if filter_type == "sobel":
        Gx = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ])
    
    elif filter_type == "schar":
        Gx = np.array([
            [-3, 0, 3],
            [-10, 0, 10],
            [-3, 0, 3]
        ])
    
    elif filter_type == "laplas":
        Gx = np.array([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ])

    Gy = Gx.T
    if type(img) == th.Tensor:
        Gx = th.Tensor(Gx).view(1, 1, 3, 3)
        Gy = th.Tensor(Gy).view(1, 1, 3, 3)
    
    

    grad_x_img = F.conv2d(img, Gx.repeat(1, 3, 1, 1))
    grad_y_img = F.conv2d(img, Gy.repeat(1, 3, 1, 1))
    grad_x_depth = F.conv2d(depth, Gx)
    grad_y_depth = F.conv2d(depth, Gy)
    
    x_term = th.abs(grad_x_depth) * th.exp(-th.abs(grad_x_img))
    y_term = th.abs(grad_y_depth) * th.exp(-th.abs(grad_y_img))

    loss = th.mean(x_term + y_term)

    return loss * temperature



def ab_loss(a_tensor: Tensor, b: Tensor):
    return th.sum(((a_tensor - 1) ** 2) - b ** 2)


def ssim(
    I1: th.Tensor, 
    I2: th.Tensor,
    kernel_size: Union[Tuple[int, int], int], 
    in_channels: int=3,
    C1: float=0.01**2, 
    C2: float=0.03**2,
    return_map: bool=False
):


    
    kernel = gauss_kernel(size=kernel_size).view(1, 1, kernel_size[0], kernel_size[1])
    kernel = kernel.repeat(1, in_channels, 1, 1)

    mu_x = F.conv2d(
        input=I1, 
        weight=kernel, 
        stride=kernel.size(-1)//2
    )
    mu_y = F.conv2d(
        input=I2, 
        weight=kernel, 
        stride=kernel.size(-1)//2
    )

    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y

    sigma_x_sq = F.conv2d(
        input=I1*I1, 
        weight=kernel, 
        stride=kernel.size(-1)//2
    ) - mu_x_sq
    sigma_y_sq = F.conv2d(
        input=I2*I2, 
        weight=kernel, 
        stride=kernel.size(-1)//2
    ) - mu_y_sq
    sigma_xy = F.conv2d(
        input=I1*I2, 
        weight=kernel, 
        stride=kernel.size(-1)//2
    ) - mu_xy


    numerator = (2*mu_xy + C1) * (2*sigma_xy + C2)
    denominator = (mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2)
    result_map = numerator /  denominator
    
    out = result_map.mean()
    if return_map:
        out = (result_map.mean(), result_map)
    
    return out
    
    



if __name__ == "__main__":

    # conv = Conv2d(3, 3, kernel_size=(3, 3), padding=1, stride=1)
    # conv1 = Conv2d(1, 1, kernel_size=(3, 3), padding=1, stride=1)
    # img = th.normal(0.0, 1.0, (10, 3, 128, 128))
    # depth = th.normal(0.0, 1.0, (10, 1, 128, 128))
    # img = conv(img)
    # depth = conv1(depth)
    # points = th.randint(0, 128, (32*32, 2))
    # loss1 = smooth_loss(img, depth, points, filter_type="schar")
    # loss1.backward()
    # print(loss1)
    kernel_size = (11, 11)
    I1 = th.normal(0.0, 1.0, (10, 3, 128, 128))
    I2 = th.normal(0.0, 1.0, (10, 3, 128, 128))
    print(ssim(I1, I2, kernel_size=kernel_size, return_map=True))
    
    
    
    
    
    

    
    
    
# kernel_size = (3, 3)
# test_img = np.random.normal(0.0, 1.0, (128, 128))
# p = (32, 45)
# print(test_img[
#     p[0] - kernel_size[0]: p[0] + kernel_size[0] + 1,
#     p[1] - kernel_size[1]: p[1] + kernel_size[1] + 1
# ].shape)
