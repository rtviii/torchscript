
from torch import Tensor
import torch

def some_rotation_op(rotation:Tensor):
  [ Rxz, Ryz, Rzz ] = rotation[:, -1]
  ...
  return Rxz


torch.jit.script(some_rotation_op, (torch.rand( 3, 3),))