import torch as pt
from torch import nn, Tensor
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from Unet_custom import UNet32x32_custom

from tqdm import tqdm
import os
import argparse
from matplotlib import pyplot as plt

def save_checkpoint(model, optimizer, file_path):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    pt.save(checkpoint, file_path)
    print(f"Checkpoint saved !")


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

class DDPM:

    def __init__(self, T:int, beta_1:float, beta_T:float, img_size:int, device:pt.device) -> None:
        self.T = T
        self.beta = pt.linspace(beta_1, beta_T, T).to(device)
        self.alpha = 1.0 - self.beta
        self.alpha_bar = pt.cumprod(self.alpha, dim=0)
        self.sigma_2 = self.beta
        self.img_size = img_size
        self.device = device

    def add_noise(self, x0:Tensor, t:Tensor) -> tuple[Tensor, Tensor]:
        B = x0.size(0)
        unit_noise = pt.randn_like(x0)
        #forward process
        noised_images = pt.sqrt(self.alpha_bar[t]).view(B,1,1,1) * x0 +  unit_noise * pt.sqrt(1.0 - self.alpha_bar[t]).view(B,1,1,1)
        return noised_images, unit_noise

    @pt.no_grad()
    def sample(self, model:nn.Module, n:int):

        model.eval()
        x = pt.randn((n, 3, self.img_size, self.img_size)).to(self.device)
        for i in tqdm(range(self.T-1, -1, -1)):
            t = (pt.ones(n) * i).long().to(self.device)
            predicted_noise = model(x, t)
            alpha = self.alpha[t][:, None, None, None]
            alpha_hat = self.alpha_bar[t][:, None, None, None]
            beta = self.beta[t][:, None, None, None]
            if i > 0:
                noise = pt.randn_like(x)
            else:
                noise = pt.zeros_like(x)
            x = 1 / pt.sqrt(alpha) * (x - ((1 - alpha) / (pt.sqrt(1 - alpha_hat))) * predicted_noise) + pt.sqrt(beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(pt.uint8)
        return x


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(pt.cat([
        pt.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def train(args):
    
    # set up DDP (distributed data parallel).
    # torchrun command sets the env variables RANK, LOCAL_RANK, and WORLD_SIZE
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        # use of DDP atm demands CUDA, we set the device appropriately according to rank
        assert pt.cuda.is_available(), "for now i think we need CUDA for DDP"
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


    ################################ STANFORD CARS

    # transforms = v2.Compose([
    #     v2.ToImage(),
    #     v2.ToDtype(pt.float32, scale=True),
    #     v2.Resize((256,256)),
    # ])


    ################################# CIFAR10
    transforms = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(pt.float32, scale=True), #[0,1]
        v2.Lambda(lambda x:x*2.0 - 1.0), # [-1,1]
    ])


    # trainset = StanfordCars(root="./", split="train", transform=transforms)
    # valset = StanfordCars(root="./", split="test", transform=transforms)
    trainset = CIFAR10(root=args.dataset_dir, download=False, train=True, transform=transforms)
    valset = CIFAR10(root=args.dataset_dir, download=False, train=False, transform=transforms)
    dataset = ConcatDataset([trainset, valset])
    # load the whole dataset on cpu memory
    labels = pt.tensor([dataset[i][1] for i in range(len(dataset))], dtype=pt.int64)
    dataset = pt.stack([dataset[i][0] for i in range(len(dataset))])

    pt.manual_seed(0)
    if pt.cuda.is_available():
        pt.cuda.manual_seed(0)

    model = UNet32x32_custom(device=device, time_dim=args.time_dim).to(device)
    if args.compile:
        model = pt.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    optimizer = pt.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    if args.from_pretrained:
        load_checkpoint(model, optimizer, args.saved_model_path)
    # diffusion process
    diffusion = DDPM(args.T, args.beta_1, args.beta_T, device)
    if master_process:
        print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    pt.set_float32_matmul_precision('high')
    T = args.T
    assert args.batch_size % ddp_world_size == 0
    # B is the batch_size per gpu while args.batch_size is the effective batch size
    B = args.batch_size // ddp_world_size
    steps_per_epoch = len(dataset) // args.batch_size
    steps = args.epochs * steps_per_epoch
    lr = args.lr
    # portion of the dataset to sample from for each GPU run. 
    # For example, with 2 GPUs: cuda:0 samples from the first half of the dataset (0 to N), cuda:1 samples from the other half (N to 2N)
    N = len(dataset) // ddp_world_size
    id_min, id_max = ddp_rank*N, N + ddp_rank*N

    model.train()
    pbar = tqdm(range(1, steps+1)) if master_process else range(1, steps+1)
        
    for step in pbar:
        
        # select batch_size random images
        random_ids = pt.randint(id_min, id_max, size=(B,))
        images = dataset[random_ids].to(device)
        # select randomly timesteps to denoise
        t = pt.randint(0, T, size=(B,)).to(device)

        noised_images, unit_noise = diffusion.add_noise(images, t)
        # make the denoiser predict the unit noise
        predicted_noise = model(noised_images, t)
        # mse loss
        loss = nn.functional.mse_loss(predicted_noise, unit_noise)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lr *= 0.9999
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # For printing, synchronize the loss across GPUs
        if ddp:
            dist.all_reduce(loss, op=dist.ReduceOp.AVG)
        if master_process:
            pbar.set_description_str(f"Epoch {step/steps_per_epoch:2f} | Step {step} | Loss: {loss.item():8f} | lr: {lr:.8f}")

    if master_process:
        save_checkpoint(model, optimizer, args.saved_model_path)
    if ddp:
        destroy_process_group()
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--T", type=int, default=1000)
    parser.add_argument("--beta_1", type=float, default=0.0001)
    parser.add_argument("--beta_T", type=float, default=0.02)
    parser.add_argument("--time_dim", type=int, default=256)

    parser.add_argument("--dataset_dir", type=str, default="./cifar10")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=256, help="effective batch size")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--saved_model_path", type=str, default="models/ddpm_cifar10.pt")
    parser.add_argument("--from_pretrained", action="store_true", default=False)
    parser.add_argument("--compile", action="store_true", default=False)
    args = parser.parse_args()
    
    train(args)