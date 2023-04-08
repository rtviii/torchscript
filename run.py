import pyro
import pyro.distributions  as distributions
from torch import tensor
import torch


beta_mean_var_to_alpha_beta=torch.jit.load("beta_mean_var_to_alpha_beta.pt")



def my_model():
    # Input to the "micelle" function
    x = torch.randn(10,2)
    # Use the "micelle" function
    y = beta_mean_var_to_alpha_beta(torch.rand(1),torch.rand(1))
    # # Define your statistical model with pyro.sample
    z = pyro.sample('z', distributions.Normal(y[0], tensor(1.0)))
    return z
# 
my_model()
