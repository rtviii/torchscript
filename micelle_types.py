from typing import Optional
import typing
import pydantic
from torch import Tensor

class Dictlike(pydantic.BaseModel):

      class Config:
          allow_population_by_field_name = False
          arbitrary_types_allowed = True
      def __getattr__(self, attr):
          return super().dict().__getattribute__(attr)
      def __getitem__(self, attr):
          return super().dict().__getitem__(attr)
      def __setitem__(self, key,val):
          return super().dict().__setitem__(key,val)

      def dict(self,):
          return super().dict()

class YamlConfig(Dictlike): 

    class Latents(Dictlike): 
      radius_circle_loc: float
      radius_circle_scale: float
      a_loc: float
      b_loc: float
      c_loc: float

      a_scale: float
      b_scale: float
      c_scale: float

      inner_shell_ratio_loc  : float
      inner_shell_ratio_scale: float

      shell_density_ratio_loc  : float
      shell_density_ratio_scale: float

      noise_loc: float
      noise_scale: float

    a_gt                  : float
    b_gt                  : float
    c_gt                  : float
    include_fixed         : bool
    inner_shell_ratio_gt  : float
    latents               : Latents
    noise_gt              : float
    radius_circle_gt      : float
    rotation_angles_deg   : typing.List[float]
    rotation_convention   : typing.Literal["ZYX","XYZ"] # <-- add other on per-need
    seed                  : int
    shell_density_ratio_gt: float
    vol_path              : str
    translation           : typing.List[float]

class SimulatedData(Dictlike):
    rotation              : Tensor
    shift_x               : Tensor
    shift_y               : Tensor
    a_gt                  : Tensor
    b_gt                  : Tensor
    c_gt                  : Tensor
    radius_circle_gt      : Tensor
    inner_shell_ratio_gt  : Tensor
    shell_density_ratio_gt: Tensor
    global_scale_gt       : Tensor
    noise_gt              : Tensor
    proj                  : Tensor
    measurement_gt        : Tensor
    x_mesh                : Optional[Tensor]
    y_mesh                : Optional[Tensor]

class ModelParams(Dictlike): 

      proj                     : Tensor
      rotation                 : Tensor
      shift_x                  : Tensor
      shift_y                  : Tensor
      global_scale_loc         : Tensor
      global_scale_scale       : Tensor
      x_mesh                   : Tensor
      y_mesh                   : Tensor
      a_loc                    : Tensor
      b_loc                    : Tensor
      c_loc                    : Tensor
      a_scale                  : Tensor
      b_scale                  : Tensor
      c_scale                  : Tensor
      radius_circle_loc        : Tensor
      radius_circle_scale      : Tensor
      inner_shell_ratio_loc    : Tensor
      inner_shell_ratio_scale  : Tensor
      shell_density_ratio_loc  : Tensor
      shell_density_ratio_scale: Tensor
      noise_scale              : Tensor
      noise_loc                : Tensor