import torch as pt
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from Unet import UNet

from tqdm import tqdm
from code import interact

def main():

    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(pt.float32, scale=True),
        v2.Lambda(lambda x:x*2.0 - 1.0)
    ])

    trainset = CIFAR10(
        root=r"C:\Users\huang\Desktop\implementation\DDPM\cifar10", 
        download=False, 
        train=True, 
        transform=transforms
    )

    valset = CIFAR10(
        root=r"C:\Users\huang\Desktop\implementation\DDPM\cifar10", 
        download=False, 
        train=False, 
        transform=transforms
    )

    dataset = ConcatDataset([trainset, valset])

    # Parameters
    channels = 128 # initial channel expansion
    heads = 4 # attention heads
    sigma_min = 0.001
    epochs = 1
    batch_size = 256
    lr = 5e-4
    device = pt.device("cuda")
    H,W = 32,32

    vector_field = UNet(channels,  heads).to(device)
    optimizer = pt.optim.Adam(vector_field.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    def Phi_cond_x1(t:Tensor, x0:Tensor, x1:Tensor) -> Tensor:
        '''t: (B,1), x0: (B,C,H,W), x1: (B,C,H,W)'''
        return t*x1 + (1.0 - (1.0 - sigma_min)*t)*x0

    def true_vf_cond_x1(x0:Tensor, x1:Tensor) -> Tensor:
        return x1 - (1.0 - sigma_min)*x0

    # Training

    progress_bar = tqdm(range(1, epochs+1))
    step = 0
    for epoch in progress_bar:

        for i, (x1, _) in enumerate(loader):

            x1 = x1.to(device)
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

            progress_bar.set_description_str(f"Epoch {epoch} | Step {step+i} | Loss: {loss.item():8f}")

        step += len(loader)

    pt.save(vector_field.state_dict(), f"models/vf_cifar10.pt")


if __name__ == "__main__":

    main()