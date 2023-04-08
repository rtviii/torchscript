import numpy
import torch
from torch import Tensor, tensor
from micelle_types import SimulatedData, YamlConfig
from scipy.spatial.transform import Rotation
from pyro import distributions 
import libtrace as geometric_micelle
import mrcfile

# this can be traced if:
# - pass in the rotation tensor (lift the scipy bit up)
# - volume as tensor

def sim_data(volume:Tensor, rotation:Tensor,proj:Tensor, translation:Tensor,
             x_mesh             : Tensor,
             y_mesh             : Tensor,
             a                  : Tensor,
             b                  : Tensor,
             c                  : Tensor,
             radius_circle      : Tensor,
             inner_shell_ratio  : Tensor,
             shell_density_ratio: Tensor,
             noise_gt           : Tensor,

             ): 

  n_pix  = len(proj)
  arr_1d = torch.arange(-n_pix // 2, n_pix // 2, step=1)

  y_mesh, x_mesh = torch.meshgrid(arr_1d, arr_1d)

  shift_x_gt, shift_y_gt, _ = rotation @ translation

  micelle_gt = geometric_micelle.two_phase_micelle(x_mesh,
                                                   y_mesh,
                                                   a                  ,
                                                   b                  ,
                                                   c                  ,
                                                   rotation,
                                                   radius_circle       ,
                                                   inner_shell_ratio   ,
                                                   shell_density_ratio ,
                                                   shift_x_gt             ,
                                                   shift_y_gt             ,
                                                   )
  micelle_gt      = micelle_gt / micelle_gt.max()
  global_scale_gt = proj.max()
  micelle_gt      = micelle_gt * global_scale_gt
  measurement_gt  = distributions.Normal(proj + micelle_gt, noise_gt).sample()

  sim_data_return = {
      'rotation'              : rotation,
      'shift_x'               : shift_x_gt,
      'shift_y'               : shift_y_gt,
      'a_gt'                  : a,
      'b_gt'                  : b,
      'c_gt'                  : c,
      'radius_circle_gt'      : radius_circle,
      'inner_shell_ratio_gt'  : inner_shell_ratio,
      'shell_density_ratio_gt': shell_density_ratio,
      'noise_gt'              : noise_gt,
      'global_scale_gt'       : global_scale_gt,
      'proj'                  : proj,
      'measurement_gt'        : measurement_gt,

      'x_mesh'                : x_mesh,
      'y_mesh'                : y_mesh,
    }

  return SimulatedData.parse_obj(sim_data_return)


volume_path = "/home/rxz/dev/physics_aware_cryoem/6dmy_molmap-3.6A-zeropad136.mrc"
with mrcfile.open(volume_path) as file: 
   vol = file.data
   
# ----------------
proj   = torch.from_numpy(vol.sum(-1)).to(device)
translation = torch.tensor(meta_data.translation).to(device)





x_mesh.to(device), 
y_mesh.to(device),
tensor(meta_data.a_gt, device=device)
tensor(meta_data.b_gt, device=device)
tensor(meta_data.c_gt, device=device)
tensor(rotation_gt, device=device)
tensor(meta_data.radius_circle_gt, device=device)
tensor(meta_data.inner_shell_ratio_gt, device=device)
tensor(meta_data.shell_density_ratio_gt, device=device)
tensor(shift_x_gt, device=device)
tensor(shift_y_gt, device=device)


_rotation  = torch.from_numpy(
 Rotation.from_euler(
    meta_data.rotation_convention,
    meta_data.rotation_angles_deg,
        degrees=True).as_matrix()).float().to(device)


noise_gt

simulated:SimulatedData = sim_data(meta_data, tensor(vol),torch_device) # TODO: save simulation data