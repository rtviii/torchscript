import io
import math
from pprint import pprint
import sys
from torch import ScriptFunction, Tensor, tensor
import torch


# ? NOTE: deativated "n_crop" in cylinder code to streamline things.



def min_max_border(n_border:Tensor, n:Tensor):
  # print(f"※------------------------------[{sys._getframe().f_code.co_name}] input shapes: ", n_border.shape, n.shape)
  min_space = tensor(1)
  n_border  = torch.min(n // 2 - min_space, n_border)
  n_border  = torch.max(min_space, n_border)
  return n_border





def fourier_to_primal_2D(f:Tensor):
    f = torch.fft.ifftshift(f, dim=(-2, -1))
    return torch.fft.fftshift(
        torch.fft.ifftn(f, s=(f.shape[-2], f.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )

def primal_to_fourier_2D(r:Tensor):
    r = torch.fft.ifftshift(r, dim=(-2, -1))
    return torch.fft.fftshift(
        torch.fft.fftn(r, s=(r.shape[-2], r.shape[-1]), dim=(-2, -1)), dim=(-2, -1)
    )

def gamma_mean_var_to_alpha_beta(mean:Tensor, var:Tensor):
  beta  = mean / var
  alpha = mean * beta
  return alpha, beta

def beta_mean_var_to_alpha_beta(mean:Tensor,var:Tensor):

  mean_inv = 1/mean
  alpha    = ((1-mean)/var - mean_inv)*mean*mean
  beta     = alpha*(mean_inv-1)
  return alpha, beta

def projected_rotated_circle(x_mesh:Tensor, y_mesh:Tensor, radius_circle:Tensor, rotation:Tensor):
  # normal_axis = rotation[:, -1]  # rotation dotted with zaxis (projection axis)
  # Tensor cannot be used as a tuple:
  nx = rotation[0,-1]
  ny = rotation[1,-1]
  nz = rotation[2,-1]

  ellipse     = (nx * nx + nz * nz) * x_mesh * x_mesh + 2 * nx * ny * x_mesh * y_mesh + (nz * nz + ny * ny) * y_mesh * y_mesh - nz * nz * radius_circle * radius_circle < 0
  return ellipse

def project_rotated_cylinder(
    x_mesh       : Tensor,
    y_mesh       : Tensor,
    radius_circle: Tensor,
    h            : Tensor,
    rotation     : Tensor,
    shift_x = tensor(0),
    shift_y = tensor(0)
    ):

  n   = x_mesh.shape[0]

  # [Rxz, Ryz, Rzz] = rotation[:, -1]
  # https://github.com/pyg-team/pytorch_geometric/issues/2099#issuecomment-1496790147
  Rxz = rotation[0, -1]
  Ryz = rotation[1, -1]
  Rzz = rotation[2, -1]

  # one, zero = tensor(1., device=DEVICE_GLOBAL), tensor(0., device=DEVICE_GLOBAL)
  # https://stackoverflow.com/a/74392764/10697358
  one, zero = tensor(1.), tensor(0.)

  if torch.isclose(Rzz, one, atol=1e-4):
    case = 'about z-axis'
    circle = projected_rotated_circle(x_mesh-shift_x, y_mesh-shift_y, radius_circle, rotation=torch.eye(3))
    fill_factor = h
    proj_cylinder = fill_factor * circle
    # print(case)

    return proj_cylinder

  elif torch.isclose(Rxz.abs(), one):  # 90 deg, line along y-axis
    case = '90 deg, line along x-axis'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = torch.round(n / 2 - h / 2)
    line[:, :n_border_x] = line[:, -n_border_x:] = 0

  elif torch.isclose(Ryz.abs(), one):  #
    case = '90 deg, line along y-axis'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = torch.round(n / 2 - h / 2)
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  elif torch.isclose(Ryz, zero) and not torch.isclose(Rzz, one):
    case = '90 deg, line along x-axis with z-tilt'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = (n / 2 - h / 2 * Rxz.abs()).round().long()
    line[:, :n_border_x] = line[:, -n_border_x:] = 0

  elif torch.isclose(Rxz, zero) and not torch.isclose(Rzz, one):
    case = '90 deg, line along y-axis with z-tilt'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = (n / 2 - h / 2 * Ryz.abs()).round().long()
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  else:
    case = 'else'
    line_test = Rxz * y_mesh - Ryz * x_mesh  # intercept zero since cylinder centered

    line_width_factor = 2 # sqrt 2 ?
    line_clipped = line_width_factor - torch.clamp(line_test.abs(), min=0, max=line_width_factor)

    n_border_x = (n / 2 - h / 2 * Rxz.abs()).round().long()
    n_border_y = (n / 2 - h / 2 * Ryz.abs()).round().long()

    
    n_border_x = min_max_border(n_border_x, tensor(n))
    n_border_y = min_max_border(n_border_y, tensor(n))

    line_clipped[:n_border_y, :] = line_clipped[-n_border_y:, :] = line_clipped[:, :n_border_x] = line_clipped[:, -n_border_x:] = 0
    line = line_clipped

  line_f    = primal_to_fourier_2D(line)
  ellipse   = projected_rotated_circle(x_mesh-shift_x, y_mesh-shift_y, radius_circle, rotation)
  ellipse_f = primal_to_fourier_2D(ellipse)
  product   = ellipse_f * line_f # TODO: add anti-aliasing for blurring

  # if n_crop is not None:

  #   [ idx_start, idx_end ] = [ n//2 - n_crop//2, n//2 + n_crop//2 ]
  #   product = product[idx_start:idx_end,idx_start:idx_end]

  convolve      = fourier_to_primal_2D(product)
  proj_cylinder = convolve.real

  return proj_cylinder

def project_rotated_ellipsoid(x_mesh:Tensor, y_mesh:Tensor, a:Tensor, b:Tensor, c:Tensor, rotation:Tensor):

  Rxx,Rxy,Rxz    = rotation[0]
  Ryx,Ryy,Ryz    = rotation[1]
  Rzx,Rzy,Rzz    = rotation[2]
  x,y            = x_mesh, y_mesh
  piece_2        = a*b*c*torch.sqrt(-Rxx**2*Ryz**2*c**2*x**2 - Rxx**2*Rzz**2*b**2*x**2 - 2*Rxx*Rxy*Ryz**2*c**2*x*y - 2*Rxx*Rxy*Rzz**2*b**2*x*y + 2*Rxx*Rxz*Ryx*Ryz*c**2*x**2 + 2*Rxx*Rxz*Ryy*Ryz*c**2*x*y + 2*Rxx*Rxz*Rzx*Rzz*b**2*x**2 + 2*Rxx*Rxz*Rzy*Rzz*b**2*x*y - Rxy**2*Ryz**2*c**2*y**2 - Rxy**2*Rzz**2*b**2*y**2 + 2*Rxy*Rxz*Ryx*Ryz*c**2*x*y + 2*Rxy*Rxz*Ryy*Ryz*c**2*y**2 + 2*Rxy*Rxz*Rzx*Rzz*b**2*x*y + 2*Rxy*Rxz*Rzy*Rzz*b**2*y**2 - Rxz**2*Ryx**2*c**2*x**2 - 2*Rxz**2*Ryx*Ryy*c**2*x*y - Rxz**2*Ryy**2*c**2*y**2 - Rxz**2*Rzx**2*b**2*x**2 - 2*Rxz**2*Rzx*Rzy*b**2*x*y - Rxz**2*Rzy**2*b**2*y**2 + Rxz**2*b**2*c**2 - Ryx**2*Rzz**2*a**2*x**2 - 2*Ryx*Ryy*Rzz**2*a**2*x*y + 2*Ryx*Ryz*Rzx*Rzz*a**2*x**2 + 2*Ryx*Ryz*Rzy*Rzz*a**2*x*y - Ryy**2*Rzz**2*a**2*y**2 + 2*Ryy*Ryz*Rzx*Rzz*a**2*x*y + 2*Ryy*Ryz*Rzy*Rzz*a**2*y**2 - Ryz**2*Rzx**2*a**2*x**2 - 2*Ryz**2*Rzx*Rzy*a**2*x*y - Ryz**2*Rzy**2*a**2*y**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  piece_3        = (Rxz**2*b**2*c**2 + Ryz**2*a**2*c**2 + Rzz**2*a**2*b**2)
  z_nans         = 2*piece_2/piece_3
  proj_ellipsoid = torch.nan_to_num(z_nans, 0)

  return proj_ellipsoid

def two_phase_micelle(
                       x_mesh             : Tensor,
                       y_mesh             : Tensor,
                       a                  : Tensor,
                       b                  : Tensor,
                       c                  : Tensor,
                       rotation           : Tensor,
                       radius_circle      : Tensor,
                       inner_shell_ratio  : Tensor,
                       shell_density_ratio: Tensor,
                       shift_x            : Tensor=tensor(0),
                       shift_y            : Tensor=tensor(0)): 

  proj_ellipsoid_outer = project_rotated_ellipsoid(x_mesh-shift_x, y_mesh-shift_y, a, b, c, rotation.T)
  proj_ellipsoid_inner = project_rotated_ellipsoid(x_mesh-shift_x, y_mesh-shift_y,
                                                   a * inner_shell_ratio,
                                                   b * inner_shell_ratio,
                                                   c * inner_shell_ratio,
                                                   rotation.T)

  shell               = proj_ellipsoid_outer - proj_ellipsoid_inner
  h_outer             = c * 2
  volume_outer        = math.pi * radius_circle ** 2 * h_outer
  proj_cylinder_outer = project_rotated_cylinder(x_mesh,
                                                 y_mesh,
                                                 radius_circle = radius_circle,
                                                 h             = h_outer,
                                                 rotation      = rotation,
                                                 shift_x       = shift_x,
                                                 shift_y       = shift_y)

  proj_cylinder_outer = volume_outer * proj_cylinder_outer / proj_cylinder_outer.sum()
  h_inner             = c * inner_shell_ratio * 2
  volume_inner        = math.pi * radius_circle ** 2 * h_inner
  proj_cylinder_inner = project_rotated_cylinder(x_mesh,
                                                 y_mesh,
                                                 radius_circle = radius_circle,
                                                 h             = h_inner,
                                                 rotation      = rotation,
                                                 shift_x       = shift_x,
                                                 shift_y       = shift_y)

  proj_cylinder_inner = volume_inner * proj_cylinder_inner / proj_cylinder_inner.sum()
  proj_cylinder_shell = proj_cylinder_outer - proj_cylinder_inner
  micelle             = shell_density_ratio * shell + proj_ellipsoid_inner - (shell_density_ratio * proj_cylinder_shell + proj_cylinder_inner)

  return micelle


trace = torch.jit.trace
script = torch.jit.script

DEVICE_GLOBAL = torch.device("cuda:0")

gamma__I1   = torch.rand(1).squeeze()
gamma__I2   = torch.rand(1).squeeze()
beta__I1    = torch.rand(1).squeeze()
beta__I2    = torch.rand(1).squeeze()
primal__I1  = torch.rand(136,136)
fourier__I1 = torch.rand(136,136)

# ※------------------------------[primal_to_fourier_2D] input shapes        : torch.Size([136, 136])
# ※------------------------------[fourier_to_primal_2D] input shapes        : torch.Size([136, 136])
# ※------------------------------[gamma_mean_var_to_alpha_beta] input shapes: torch.Size([]) torch.Size([])
# ※------------------------------[beta_mean_var_to_alpha_beta] input shapes : torch.Size([]) torch.Size([])
TRC_fourier_to_primal_2D         = trace(fourier_to_primal_2D, (fourier__I1))
TRC_primal_to_fourier_2D         = trace(primal_to_fourier_2D, (primal__I1))

TRC_gamma_mean_var_to_alpha_beta = trace(gamma_mean_var_to_alpha_beta, (gamma__I1, gamma__I2))
TRC_beta_mean_var_to_alpha_beta  = trace(beta_mean_var_to_alpha_beta, (beta__I1, beta__I2))

projected_ellipsoid__I1 = torch.rand(136,136)
projected_ellipsoid__I2 = torch.rand(136,136)
projected_ellipsoid__I3 = torch.rand(1).squeeze()
projected_ellipsoid__I4 = torch.rand(1).squeeze()
projected_ellipsoid__I5 = torch.rand(1).squeeze()
projected_ellipsoid__I6 = torch.rand(3,3)
# ※------------------------------[project_rotated_ellipsoid] input shapes   : torch.Size([136, 136]) torch.Size([136, 136]) torch.Size([]) torch.Size([]) torch.Size([]) torch.Size([3, 3])
TRC_project_rotated_ellipsoid = trace(project_rotated_ellipsoid, (projected_ellipsoid__I1, projected_ellipsoid__I2, projected_ellipsoid__I3, projected_ellipsoid__I4, projected_ellipsoid__I5, projected_ellipsoid__I6))

projected_circle__I1 = torch.rand(136,136)
projected_circle__I2 = torch.rand(136,136)
projected_circle__I3 = torch.rand(1).squeeze()
projected_circle__I4 = torch.rand(3,3)
# ※------------------------------[projected_rotated_circle] input shapes:  torch.Size([136, 136]) torch.Size([136, 136]) torch.Size([]) torch.Size([3, 3])
TRC_projected_rotated_circle  = trace(projected_rotated_circle, (projected_circle__I1, projected_circle__I2, projected_circle__I3, projected_circle__I4))









torch.jit.save(TRC_fourier_to_primal_2D        , "TRC_fourier_to_primal_2D.pt"        )
torch.jit.save(TRC_primal_to_fourier_2D        , "TRC_primal_to_fourier_2D.pt"        )
torch.jit.save(TRC_gamma_mean_var_to_alpha_beta, "TRC_gamma_mean_var_to_alpha_beta.pt")
torch.jit.save(TRC_beta_mean_var_to_alpha_beta , "TRC_beta_mean_var_to_alpha_beta.pt" )
torch.jit.save(TRC_project_rotated_ellipsoid   , "TRC_project_rotated_ellipsoid.pt"   )
torch.jit.save(TRC_projected_rotated_circle    , "TRC_projected_rotated_circle.pt"    )

GLOBAL_DEVICE='cpu'

with open ("TRC_fourier_to_primal_2D.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_fourier_to_primal_2D         = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_primal_to_fourier_2D.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_primal_to_fourier_2D         = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_gamma_mean_var_to_alpha_beta.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_gamma_mean_var_to_alpha_beta = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_beta_mean_var_to_alpha_beta.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_beta_mean_var_to_alpha_beta  = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_project_rotated_ellipsoid.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_project_rotated_ellipsoid    = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_projected_rotated_circle.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_projected_rotated_circle     = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)
with open ("TRC_project_rotated_cylinder.pt", 'rb') as f:
    buffer = io.BytesIO(f.read())
    TRC_project_rotated_cylinder     = torch.jit.load(buffer, map_location=GLOBAL_DEVICE)

@torch.jit.script
def script_project_rotated_cylinder(
    x_mesh       : Tensor,
    y_mesh       : Tensor,
    radius_circle: Tensor,
    h            : Tensor,
    rotation     : Tensor,
    shift_x = tensor(0),
    shift_y = tensor(0)
    ):

  n   = x_mesh.shape[0]

  # [Rxz, Ryz, Rzz] = rotation[:, -1]
  # https://github.com/pyg-team/pytorch_geometric/issues/2099#issuecomment-1496790147
  Rxz = rotation[0, -1]
  Ryz = rotation[1, -1]
  Rzz = rotation[2, -1]

  # one, zero = tensor(1., device=DEVICE_GLOBAL), tensor(0., device=DEVICE_GLOBAL)
  # https://stackoverflow.com/a/74392764/10697358
  one, zero = tensor(1.), tensor(0.)

  if torch.isclose(Rzz, one, atol=1e-4):
    case = 'about z-axis'
    circle = projected_rotated_circle(x_mesh-shift_x, y_mesh-shift_y, radius_circle, rotation=torch.eye(3))
    fill_factor = h
    proj_cylinder = fill_factor * circle
    # print(case)

    return proj_cylinder

  elif torch.isclose(Rxz.abs(), one):  # 90 deg, line along y-axis
    case = '90 deg, line along x-axis'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = torch.round(n / 2 - h / 2)
    line[:, :n_border_x] = line[:, -n_border_x:] = 0

  elif torch.isclose(Ryz.abs(), one):  #
    case = '90 deg, line along y-axis'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = torch.round(n / 2 - h / 2)
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  elif torch.isclose(Ryz, zero) and not torch.isclose(Rzz, one):
    case = '90 deg, line along x-axis with z-tilt'
    line = torch.zeros(n, n)
    line[n // 2, :] = 1
    n_border_x = (n / 2 - h / 2 * Rxz.abs()).round().long()
    line[:, :n_border_x] = line[:, -n_border_x:] = 0

  elif torch.isclose(Rxz, zero) and not torch.isclose(Rzz, one):
    case = '90 deg, line along y-axis with z-tilt'
    line = torch.zeros(n, n)
    line[:, n // 2] = 1
    n_border_y = (n / 2 - h / 2 * Ryz.abs()).round().long()
    line[:n_border_y, :] = line[-n_border_y:, :] = 0

  else:
    case = 'else'
    line_test = Rxz * y_mesh - Ryz * x_mesh  # intercept zero since cylinder centered

    line_width_factor = 2 # sqrt 2 ?
    line_clipped = line_width_factor - torch.clamp(line_test.abs(), min=0, max=line_width_factor)

    n_border_x = (n / 2 - h / 2 * Rxz.abs()).round().long()
    n_border_y = (n / 2 - h / 2 * Ryz.abs()).round().long()

    
    n_border_x = min_max_border(n_border_x, tensor(n))
    n_border_y = min_max_border(n_border_y, tensor(n))

    line_clipped[:n_border_y, :] = line_clipped[-n_border_y:, :] = line_clipped[:, :n_border_x] = line_clipped[:, -n_border_x:] = 0
    line = line_clipped

  line_f    = primal_to_fourier_2D(line)
  ellipse   = projected_rotated_circle(x_mesh-shift_x, y_mesh-shift_y, radius_circle, rotation)
  ellipse_f = primal_to_fourier_2D(ellipse)
  product   = ellipse_f * line_f # TODO: add anti-aliasing for blurring

  # if n_crop is not None:

  #   [ idx_start, idx_end ] = [ n//2 - n_crop//2, n//2 + n_crop//2 ]
  #   product = product[idx_start:idx_end,idx_start:idx_end]

  convolve      = fourier_to_primal_2D(product)
  proj_cylinder = convolve.real

  return proj_cylinder





projected_cylinder__I1 = torch.rand(136,136)
projected_cylinder__I2 = torch.rand(136,136)
projected_cylinder__I3 = torch.rand(1).squeeze()
projected_cylinder__I4 = torch.rand(1).squeeze()
projected_cylinder__I5 = torch.rand(3,3)
# ※------------------------------[project_rotated_cylinder] input shapes    : torch.Size([136, 136]) torch.Size([136, 136]) torch.Size([]) torch.Size([]) torch.Size([3, 3])
SCRIPT_project_rotated_cylinder  = torch.jit.script(script_project_rotated_cylinder, (
    projected_cylinder__I1,
    projected_cylinder__I2,
    projected_cylinder__I3,
    projected_cylinder__I4,
    projected_cylinder__I5))



print(TRC_primal_to_fourier_2D.graph)
print(TRC_primal_to_fourier_2D.code)


torch.jit.save(SCRIPT_project_rotated_cylinder    , "SCRIPT_project_rotated_cylinder.pt"    )