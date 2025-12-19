import os, csv, argparse, torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy import linalg

from src.models.lenet import LeNet

@torch.no_grad()
def compute_real_stats(lenet, loader, device):
    feats = []
    for x, _ in tqdm(loader, desc="Real features"):
        x = x.to(device)
        _, f = lenet(x, return_features=True)
        feats.append(f.cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    return mu, cov

@torch.no_grad()
def compute_gen_stats_and_acc(lenet, gen_pt_path, device):
    blob = torch.load(gen_pt_path, map_location="cpu")
    x = blob["images"]            # [N,1,28,28] in [0,1]
    y_int = blob["labels_intended"]

    # normalize like LeNet training
    x = (x - 0.1307) / 0.3081

    loader = DataLoader(list(zip(x, y_int)), batch_size=512, shuffle=False)
    feats = []
    correct = 0
    total = 0
    for xb, yb in tqdm(loader, desc=f"Gen stats {os.path.basename(gen_pt_path)}"):
        xb = xb.to(device)
        yb = yb.to(device)
        logits, f = lenet(xb, return_features=True)
        pred = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total += yb.numel()
        feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False)
    acc = correct / max(total, 1)
    return mu, cov, acc, total

def frechet_distance(mu1, cov1, mu2, cov2, eps=1e-6):
    mu1 = np.atleast_1d(mu1); mu2 = np.atleast_1d(mu2)
    cov1 = np.atleast_2d(cov1); cov2 = np.atleast_2d(cov2)

    diff = mu1 - mu2
    # jitter for numerical stability
    cov1 = cov1 + np.eye(cov1.shape[0]) * eps
    cov2 = cov2 + np.eye(cov2.shape[0]) * eps

    covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff.dot(diff) + np.trace(cov1 + cov2 - 2.0 * covmean))

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default="/content/drive/MyDrive/CS810")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load LeNet
    lenet_ckpt = torch.load(os.path.join(args.out_root, "checkpoints/lenet.pt"), map_location=device)
    lenet = LeNet().to(device)
    lenet.load_state_dict(lenet_ckpt["model"])
    lenet.eval()

    # Real MNIST test features (cache)
    cache_path = os.path.join(args.out_root, "results/metrics/real_test_stats.npz")
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    test_loader = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=2, pin_memory=True)

    if os.path.exists(cache_path):
        npz = np.load(cache_path)
        mu_r, cov_r = npz["mu"], npz["cov"]
        print("Loaded cached real stats:", cache_path)
    else:
        mu_r, cov_r = compute_real_stats(lenet, test_loader, device)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez(cache_path, mu=mu_r, cov=cov_r)
        print("Saved real stats:", cache_path)

    runs = [
        ("VAE",  "beta=1.0",  os.path.join(args.out_root, "results/metrics/gen_vae_beta1.pt"),  "vae_beta1"),
        ("VAE",  "beta=0.5",  os.path.join(args.out_root, "results/metrics/gen_vae_beta05.pt"), "vae_beta05"),
        ("DDPM", "T=100",     os.path.join(args.out_root, "results/metrics/gen_ddpm_T100.pt"),  "ddpm_T100"),
        ("DDPM", "T=400",     os.path.join(args.out_root, "results/metrics/gen_ddpm_T400.pt"),  "ddpm_T400"),
    ]

    out_csv = os.path.join(args.out_root, "results/metrics/metrics.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "setting", "run", "n_gen", "lenet_acc_intended", "feature_frechet"])

        for model_name, setting, path, run in runs:
            mu_g, cov_g, acc, n = compute_gen_stats_and_acc(lenet, path, device)
            fd = frechet_distance(mu_r, cov_r, mu_g, cov_g)
            w.writerow([model_name, setting, run, n, acc, fd])
            print(f"{run}: acc={acc:.4f}  fFrechet={fd:.3f}  n={n}")

    print("Wrote:", out_csv)

if __name__ == "__main__":
    main()