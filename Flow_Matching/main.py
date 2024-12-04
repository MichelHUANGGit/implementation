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

def save_checkpoint(model, optimizer, step, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'step': step,
    }
    pt.save(checkpoint, file_path)
    print(f"Checkpoint saved at step {step}.")


def load_checkpoint(model:nn.Module, optimizer:pt.optim.Adam, file_path):
    checkpoint = pt.load(file_path)
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # Remove the '_orig_mod.' prefix
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']


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
    lr = args.lr
    num_workers = args.num_workers
    device = pt.device("cuda")
    H,W = 32,32
    
    dataloader = DataLoader(dataset, batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    vector_field = UNet(channels, heads).to(device)
    print(f"Number of parameters {sum(p.numel() for p in vector_field.parameters()):,}")
    optimizer = pt.optim.Adam(vector_field.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    if args.from_pretrained:
        _ = load_checkpoint(vector_field, optimizer, args.saved_model_path)
    if args.compile:
        vector_field = pt.compile(vector_field)

    def Phi_cond_x1(t:Tensor, x0:Tensor, x1:Tensor) -> Tensor:
        '''t: (B,1), x0: (B,C,H,W), x1: (B,C,H,W)'''
        return t*x1 + (1.0 - (1.0 - sigma_min)*t)*x0

    def true_vf_cond_x1(x0:Tensor, x1:Tensor) -> Tensor:
        return x1 - (1.0 - sigma_min)*x0

    # Training

    pt.set_float32_matmul_precision("medium")
    progress_bar = tqdm(range(1, epochs+1))
    step = 0
    for epoch in progress_bar:
        
        for i, x1 in enumerate(dataloader, start=1):
            batch_size = x1.size(0)
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
        step += i

    save_checkpoint(vector_field, optimizer, step, args.saved_model_path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--saved_model_path", type=str, default="models/vf_cifar10.pt")
    parser.add_argument("--from_pretrained", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--sigma_min", type=float, default=0.001)

    args = parser.parse_args()
    main(args)