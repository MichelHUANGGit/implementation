import torch as pt
from torch import nn, Tensor
from torchvision.transforms import v2
from Unet import UNet
from torchdiffeq import odeint

import numpy as np
import argparse


def sample(args):

    vector_field = UNet(256, 4)
    checkpoint = pt.load(args.model_path)
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # Remove the '_orig_mod.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    del checkpoint
    #Trained vector field
    vector_field.load_state_dict(new_state_dict)




    # Sampling
    device = pt.device("cuda")
    vector_field.eval().to(device)
    class ODE_VF(nn.Module):

        @pt.no_grad()
        def forward(self, t:Tensor, x:Tensor) -> Tensor:
            batch_size = x.size(0)
            t_expanded = t.expand(size=(batch_size, 1))

            return vector_field(x, t_expanded)
        
    ode_vf_func = ODE_VF()    

    x0 = pt.randn(size=(args.batch_size, 3, 32, 32), device=device)
    t = pt.linspace(0.0, 1.0, 100, device=device)
    solution = odeint(ode_vf_func, x0, t)
    samples = solution[-1]

    images_tensor = (samples.cpu().clip(-1.0, 1.0) + 1.0) / 2.0
    pt.save(images_tensor, args.saved_images_path)

    return images_tensor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--saved_images_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    args = parser.parse_args()

    sampled_images = sample(args)

