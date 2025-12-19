import os, argparse, torch
from tqdm import tqdm

from src.models.vae import ConditionalVAE
from src.models.unet_mnist import TinyConditionalUNet
from src.models.ddpm import DDPM

def gen_vae(ckpt_path, out_path, n_per_class=1000, z_dim=20, hidden_dim=256, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    args = ckpt.get("args", {})
    z_dim = args.get("z_dim", z_dim)
    hidden_dim = args.get("hidden_dim", hidden_dim)

    model = ConditionalVAE(z_dim=z_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ys = torch.arange(10, device=device).repeat_interleave(n_per_class)
    z = torch.randn(ys.size(0), z_dim, device=device)
    with torch.no_grad():
        logits = model.decode(z, ys)
        x = torch.sigmoid(logits).view(-1, 1, 28, 28).clamp(0, 1)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"images": x.cpu(), "labels_intended": ys.cpu()}, out_path)
    print("Saved:", out_path, "images:", x.shape)

def gen_ddpm(ckpt_path, out_path, n_per_class=1000, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get("args", {})
    T = int(cfg["T"])
    base_ch = int(cfg.get("base_ch", 64))
    beta_start = float(cfg.get("beta_start", 1e-4))
    beta_end = float(cfg.get("beta_end", 0.02))

    model = TinyConditionalUNet(in_ch=1, base_ch=base_ch).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(T=T, beta_start=beta_start, beta_end=beta_end, device=device)

    ys = torch.arange(10, device=device).repeat_interleave(n_per_class)

    # sample in chunks to avoid OOM
    chunks = []
    bs = 256
    with torch.no_grad():
        for i in tqdm(range(0, ys.size(0), bs), desc=f"Sampling DDPM T={T}"):
            yb = ys[i:i+bs]
            xb = ddpm.sample(model, n=yb.size(0), y=yb, shape=(1,28,28))
            xb = ((xb + 1.0) / 2.0).clamp(0, 1)  # to [0,1]
            chunks.append(xb.cpu())
    x = torch.cat(chunks, dim=0)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save({"images": x, "labels_intended": ys.cpu()}, out_path)
    print("Saved:", out_path, "images:", x.shape)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_root", type=str, default="/content/drive/MyDrive/CS810")
    p.add_argument("--n_per_class", type=int, default=1000)
    p.add_argument("--device", type=str, default="cuda")
    args = p.parse_args()

    out_dir = os.path.join(args.out_root, "results/metrics")
    os.makedirs(out_dir, exist_ok=True)

    # VAE runs
    gen_vae(os.path.join(args.out_root, "checkpoints/vae_beta1.pt"),
            os.path.join(out_dir, "gen_vae_beta1.pt"),
            n_per_class=args.n_per_class, device=args.device)

    gen_vae(os.path.join(args.out_root, "checkpoints/vae_beta05.pt"),
            os.path.join(out_dir, "gen_vae_beta05.pt"),
            n_per_class=args.n_per_class, device=args.device)

    # DDPM runs
    gen_ddpm(os.path.join(args.out_root, "checkpoints/ddpm_T100.pt"),
             os.path.join(out_dir, "gen_ddpm_T100.pt"),
             n_per_class=args.n_per_class, device=args.device)

    gen_ddpm(os.path.join(args.out_root, "checkpoints/ddpm_T400.pt"),
             os.path.join(out_dir, "gen_ddpm_T400.pt"),
             n_per_class=args.n_per_class, device=args.device)

if __name__ == "__main__":
    main()