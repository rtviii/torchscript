import io
import torch
from torch import Tensor, tensor

def min_max_border(n_border:Tensor, n:Tensor):
  # print(f"※------------------------------[{sys._getframe().f_code.co_name}] input shapes: ", n_border.shape, n.shape)
  min_space = tensor(1)
  n_border  = torch.min(n // 2 - min_space, n_border)
  n_border  = torch.max(min_space, n_border)
  return n_border
