import torch as pt
from torch import nn, Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from Unet import Unet, conditional_Unet

import os
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
    checkpoint = pt.load(file_path, weights_only=True)
    new_state_dict = {}
    for key, value in checkpoint['model_state_dict'].items():
        if key.startswith('_orig_mod.'):
            new_key = key[len('_orig_mod.'):]  # Remove the '_orig_mod.' prefix
            new_state_dict[new_key] = value
        elif not(key.startswith("module.")):
            new_key = "module." + key
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value
    model.load_state_dict(new_state_dict, strict=True)
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['step']


def main(args):

    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert pt.cuda.is_available()
        init_process_group(backend='nccl')
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        pt.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    else:
        # vanilla, non-DDP run
        ddp_rank = 0
        ddp_local_rank = 0
        ddp_world_size = 1
        master_process = True
        # attempt to autodetect device
        device = "cpu"
        if pt.cuda.is_available():
            device = "cuda"
        print(f"using device: {device}")
    
    pt.manual_seed(0)
    if pt.cuda.is_available():
        pt.cuda.manual_seed(0)

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
    images = pt.stack([dataset[i][0] for i in range(len(dataset))])
    labels = pt.tensor([dataset[i][1] for i in range(len(dataset))])
    dataset = TensorDataset(images, labels)

    # Parameters
    channels = args.channels # initial channel expansion
    heads = args.heads # attention heads
    sigma_min = args.sigma_min # small variance parameter of gaussian mixture
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    num_workers = args.num_workers
    device = pt.device("cuda")
    H,W = 32,32
    
    # Initialization of the model, dataloader, optimizer
    dataloader = DataLoader(dataset, batch_size//ddp_world_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    vector_field = conditional_Unet(channels, heads, num_classes=10) if args.class_condition else Unet(channels, heads)
    vector_field.to(device)
    optimizer = pt.optim.Adam(vector_field.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)

    if args.compile:
        vector_field = pt.compile(vector_field)
    if ddp:
        vector_field = DDP(vector_field, device_ids=[ddp_local_rank])
    if args.from_pretrained:
        _ = load_checkpoint(vector_field, optimizer, args.saved_model_path)
        if master_process:
            print("Loaded checkpoint!")

    if master_process:
        print(f"Number of parameters {sum(p.numel() for p in vector_field.parameters()):,}")


    # Utility functions
    def Phi_cond_x1(t:Tensor, x0:Tensor, x1:Tensor) -> Tensor:
        '''t: (B,1), x0: (B,C,H,W), x1: (B,C,H,W)'''
        return t*x1 + (1.0 - (1.0 - sigma_min)*t)*x0

    def true_vf_cond_x1(x0:Tensor, x1:Tensor) -> Tensor:
        return x1 - (1.0 - sigma_min)*x0


    # Training

    pt.set_float32_matmul_precision("medium")
    progress_bar = tqdm(range(0, epochs+1, ddp_world_size))
    step = 0
    for epoch in progress_bar:
        
        for i, (x1, y1) in enumerate(dataloader, start=1):

            # Sampling the noise, timesteps, and data samples
            batch_size = x1.size(0)
            x1 = x1.to(device) #data samples
            y1 = y1.to(device) if args.class_condition else None # labels
            t = pt.rand(size=(batch_size, 1), device=device) #timesteps
            x0 = pt.randn(size=(batch_size, 3, H, W), device=device) #noise

            # Model prediction
            xt = Phi_cond_x1(t[..., None, None], x0, x1)
            predicted_vector_field = vector_field(xt, t, y1)
            target_vector_field = true_vf_cond_x1(x0, x1)

            # Loss
            loss = ((predicted_vector_field - target_vector_field)**2)\
                        .sum(dim=(1,2,3))\
                        .mean(0)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Exponential lr decay
            lr *= 0.9995
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # For printing, synchronize the loss across GPUs
            if ddp:
                dist.all_reduce(loss, op=dist.ReduceOp.AVG)
            if master_process:
                progress_bar.set_description_str(f"Epoch {epoch} | Step {step+i} | Loss: {loss.item():8f}")
        step += i * ddp_world_size

        if master_process and epoch%50==0 :
            save_checkpoint(vector_field, optimizer, step, args.saved_model_path)
    if ddp:
        destroy_process_group()

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, default="./cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--saved_model_path", type=str, default="models/cifar10_vf.pt")
    parser.add_argument("--from_pretrained", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    parser.add_argument("--channels", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--class_condition", action="store_true", default=False)
    parser.add_argument("--sigma_min", type=float, default=0.001)

    args = parser.parse_args()
    main(args)

    # on local
    # python main.py --epochs 1 --batch_size 16 --saved_model_path models/cifar10_vf_cond.pt --dataset_dir C:\Users\huang\Desktop\implementation\DDPM\cifar10 --from_pretrained --class_condition
    
    # on instance
    # torchrun --standalone --nproc_per_node 2 main.py --epochs 100 --batch_size 256 --saved_model_path models/cifar10_vf_cond.pt --from_pretrained --class_condition