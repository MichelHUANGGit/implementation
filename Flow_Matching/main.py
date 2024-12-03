import torch as pt
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from Unet import UNet

from tqdm import tqdm
from code import interact
import argparse


def main(args):

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(pt.float32, scale=True),
        v2.Lambda(lambda x:x*2.0 - 1.0)
    ])

    trainset = CIFAR10(
        root=args.dataset_dir, 
        download=True, 
        train=True, 
        transform=transforms
    )

    valset = CIFAR10(
        root=args.dataset_dir, 
        download=True, 
        train=False, 
        transform=transforms
    )

    dataset = ConcatDataset([trainset, valset])
    dataset = pt.stack([dataset[i][0] for i in range(len(dataset))])

    # Parameters
    channels = args.channels # initial channel expansion
    heads = args.heads # attention heads
    sigma_min = args.sigma_min
    epochs = args.epochs
    batch_size = args.batch_size
    steps_per_epochs = len(dataset) / batch_size
    steps = round(epochs * steps_per_epochs)
    lr = args.lr
    device = pt.device("cuda")
    H,W = 32,32

    vector_field = UNet(channels,  heads).to(device)
    optimizer = pt.optim.Adam(vector_field.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    if args.compile:
        vector_field = pt.compile(vector_field)

    def Phi_cond_x1(t:Tensor, x0:Tensor, x1:Tensor) -> Tensor:
        '''t: (B,1), x0: (B,C,H,W), x1: (B,C,H,W)'''
        return t*x1 + (1.0 - (1.0 - sigma_min)*t)*x0

    def true_vf_cond_x1(x0:Tensor, x1:Tensor) -> Tensor:
        return x1 - (1.0 - sigma_min)*x0

    # Training

    pt.set_float32_matmul_precision("medium")
    progress_bar = tqdm(range(1, steps+1))
    for step in progress_bar:
        
        x1_indices = pt.randint(0, len(dataset), size=(batch_size,))
        x1 = dataset[x1_indices].to(device)
        t = pt.rand(size=(batch_size, 1), device=device)
        x0 = pt.randn(size=(batch_size, 3, H, W), device=device)
        x_t_cond_x1 = Phi_cond_x1(t[..., None, None], x0, x1)
        predicted_vector_field = vector_field(x_t_cond_x1, t)
        target_vector_field = true_vf_cond_x1(x0, x1)
        loss = ((predicted_vector_field - target_vector_field)**2)\
                    .sum(dim=(1,2,3))\
                    .mean(0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.set_description_str(f"Epoch {step/steps_per_epochs:2f} | Step {step} | Loss: {loss.item():8f}")

    pt.save(vector_field.state_dict(), args.saved_model_path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--saved_model_path", type=str, default="models/vf_cifar10.pt")
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sigma_min", type=float, default=0.001)

    args = parser.parse_args()
    main(args)