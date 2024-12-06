import torch as pt
from torch import nn, Tensor
from torchvision.transforms import v2

from matplotlib import pyplot as plt
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.models import Inception_V3_Weights, inception_v3
import numpy as np
import argparse
from scipy.linalg import sqrtm
from tqdm import tqdm



def plot_sampled_images(sampled_images:Tensor, nrows:int, ncols:int, figsize:tuple):
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)  # Adjust figsize as needed
    # choose randomly nrows * ncols to plot
    random_ids = np.random.choice(np.arange(sampled_images.size(0)), size=(nrows*ncols,), replace=False)
    axes = axes.flatten()

    toPIL = v2.ToPILImage()

    for i, ax in enumerate(axes):
        random_id = random_ids[i]
        ax.imshow(toPIL(sampled_images[random_id]))
        ax.axis('off')  # Turn off the axis

    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.1)

def compute_and_save_cifar10_stats(dataset_dir, batch_size, device):
    cifar10_preprocess = v2.Compose([
        v2.ToImage(),
        v2.Resize((299,299)),
        v2.ToDtype(pt.float32, scale=True),
        v2.Lambda(lambda x:x*2.0-1.0),
    ])

    trainset = CIFAR10(
        root=dataset_dir, 
        download=False, 
        train=True, 
        transform=cifar10_preprocess
    )

    valset = CIFAR10(
        root=dataset_dir, 
        download=False, 
        train=False, 
        transform=cifar10_preprocess
    )

    dataset = ConcatDataset([trainset, valset])
    loader = DataLoader(dataset, batch_size=batch_size)

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception.dropout = nn.Identity()
    inception.fc = nn.Identity()
    inception.eval().to(device)

    N = len(dataset)
    X1 = pt.empty(size=(N, 2048))

    for i, (batch, _) in tqdm(enumerate(loader)):
        batch = batch.to(device)
        start = i * batch_size
        end = min(start + batch_size, N)
        with pt.no_grad():
            X1[start:end] = inception(batch).cpu()

    mu1 = pt.mean(X1, dim=0)
    cov1 = pt.cov(X1.T)
    pt.save(cov1, "cifar10/FID_precomputed_cov.pt")
    pt.save(mu1, "cifar10/FID_precomputed_mean.pt")
    return mu1, cov1


def compute_samples_stats(samples:Tensor, batch_size:int, device):

    loader = DataLoader(samples, batch_size=batch_size)
    resize = v2.Resize((299,299))

    inception = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    inception.dropout = nn.Identity()
    inception.fc = nn.Identity()
    inception.eval().to(device)

    N = samples.size(0)
    X2 = pt.empty(size=(N, 2048))

    for i, batch in tqdm(enumerate(loader)):
        batch = resize(batch).to(device)
        start = i * batch_size
        end = min(start + batch_size, N)
        with pt.no_grad():
            X2[start:end] = inception(batch).cpu()
    
    mu2 = pt.mean(X2, dim=0)
    cov2 = pt.cov(X2.T)

    return mu2, cov2

def compute_FID(mean1:Tensor, cov1:Tensor, mean2:Tensor, cov2:Tensor):

    cov1 = cov1.numpy()
    cov2 = cov2.numpy()
    mean1 = mean1.numpy()
    mean2 = mean2.numpy()

    mean_diff = ((mean1 - mean2)**2).sum()
    covmean, _ = sqrtm(cov1 @ cov2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    return mean_diff + np.trace(cov1 + cov2 - 2*covmean)

def main(args):

    if args.use_precomputed_stats:
        print("Using precomputed dataset mean and covariance...")
        mean1 = pt.load(args.precomputed_mean_path, weights_only=True)
        cov1 = pt.load(args.precomputed_cov_path, weights_only=True)
    else:
        print("Computing dataset mean and covariance...")
        mean1, cov1 = compute_and_save_cifar10_stats(args.dataset_dir, args.batch_size, args.device)
        print("Done !")

    sampled_images = pt.load(args.sampled_images_path, weights_only=True)
    print(f"Computing samples stats ({sampled_images.size(0)} samples)")
    mean2, cov2 = compute_samples_stats(sampled_images, args.batch_size, args.device)
    print("Done !")

    FID = compute_FID(mean1, cov1, mean2, cov2)
    print(f"FID score: {FID:.4f}")
    return FID

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sampled_images_path", type=str, required=True)
    parser.add_argument("--dataset_dir", type=str, default="cifar10/")
    parser.add_argument("--use_precomputed_stats", action="store_true", required=True, default=True, help="Whether or not to use precomputed dataset mean and covariance")
    parser.add_argument("--precomputed_mean_path", type=str, required=False, default="cifar10/FID_precomputed_mean.pt")
    parser.add_argument("--precomputed_cov_path", type=str, required=False, default="cifar10/FID_precomputed_cov.pt")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    args.device = pt.device(args.device)

    main(args)

    #python FID_score.py --sampled_images_path images/sampled_images_400epoch.pt --use_precomputed_stats --precomputed_mean_path cifar10/FID_precomputed_mean.pt --precomputed_cov_path cifar10/FID_precomputed_cov.pt

