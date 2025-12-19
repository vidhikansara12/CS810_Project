import torch
import torch.nn.functional as F
import math

def make_beta_schedule(T, beta_start=1e-4, beta_end=0.02, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

class DDPM:
    """
    DDPM helper: forward noising and reverse sampling formulas.
    Model predicts epsilon.
    """
    def __init__(self, T, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.T = T
        self.betas = make_beta_schedule(T, beta_start, beta_end, device=device)
        self.alphas = 1.0 - self.betas
        self.alphabars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphabars = torch.sqrt(self.alphabars)
        self.sqrt_one_minus_alphabars = torch.sqrt(1.0 - self.alphabars)

        # for sampling
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_var = self.betas * (1.0 - torch.cat([torch.tensor([1.0], device=device), self.alphabars[:-1]])) / (1.0 - self.alphabars)

    def q_sample(self, x0, t, noise):
        """
        x_t = sqrt(alphabar_t)*x0 + sqrt(1-alphabar_t)*noise
        t: (B,) in [0, T-1]
        """
        B = x0.size(0)
        s1 = self.sqrt_alphabars[t].view(B, 1, 1, 1)
        s2 = self.sqrt_one_minus_alphabars[t].view(B, 1, 1, 1)
        return s1 * x0 + s2 * noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t, y):
        """
        One reverse step: x_{t-1} from x_t
        """
        B = x_t.size(0)
        t_in = torch.full((B,), t, device=self.device, dtype=torch.long)

        eps_hat = model(x_t, t_in, y)

        beta_t = self.betas[t]
        sqrt_one_minus_ab = self.sqrt_one_minus_alphabars[t]
        sqrt_recip_alpha = self.sqrt_recip_alphas[t]

        # mean formula
        mean = sqrt_recip_alpha * (x_t - beta_t / sqrt_one_minus_ab * eps_hat)

        if t == 0:
            return mean

        var = self.posterior_var[t]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, n, y, shape=(1,28,28)):
        x = torch.randn((n, *shape), device=self.device)
        for t in reversed(range(self.T)):
            x = self.p_sample(model, x, t, y)
        return x
''')

# ---------- train_ddpm.py ----------
(REPO_DIR / "train_ddpm.py").write_text(r'''
import os, csv, argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.unet_mnist import TinyConditionalUNet
from src.models.ddpm import DDPM

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--T", type=int, default=100)
    p.add_argument("--beta_start", type=float, default=1e-4)
    p.add_argument("--beta_end", type=float, default=0.02)
    p.add_argument("--base_ch", type=int, default=64)

    p.add_argument("--run_name", type=str, default="ddpm_T100")
    p.add_argument("--out_root", type=str, default="/content/drive/MyDrive/CS810")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    ckpt_path = os.path.join(args.out_root, f"checkpoints/{args.run_name}.pt")
    log_csv   = os.path.join(args.out_root, f"results/logs/{args.run_name}_train.csv")
    sample_dir = os.path.join(args.out_root, "results/samples")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # For diffusion it's common to scale to [-1, 1]
    tfm = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2.0 - 1.0)])
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    model = TinyConditionalUNet(in_ch=1, base_ch=args.base_ch).to(device)
    ddpm = DDPM(T=args.T, beta_start=args.beta_start, beta_end=args.beta_end, device=device)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "mse_loss"])

    best = 1e9

    @torch.no_grad()
    def save_samples(epoch):
        model.eval()
        ys = torch.arange(10, device=device).repeat_interleave(16)  # 160 samples
        x = ddpm.sample(model, n=ys.size(0), y=ys, shape=(1,28,28))
        x = (x + 1.0) / 2.0  # back to [0,1]
        out = os.path.join(sample_dir, f"{args.run_name}_samples_ep{epoch:03d}.png")
        save_image(x, out, nrow=16)
        return out

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()

        for x0, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x0, y = x0.to(device), y.to(device)
            B = x0.size(0)
            t = torch.randint(0, args.T, (B,), device=device, dtype=torch.long)
            noise = torch.randn_like(x0)
            x_t = ddpm.q_sample(x0, t, noise)

            noise_hat = model(x_t, t, y)
            loss = F.mse_loss(noise_hat, noise)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            running += loss.item() * B
            n += B

        mse_ep = running / n
        dt = time.time() - t0

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, mse_ep])

        print(f"Epoch {epoch}: mse={mse_ep:.6f} time={dt:.1f}s")

        if epoch == 1 or epoch % 2 == 0 or epoch == args.epochs:
            s = save_samples(epoch)
            print("Saved:", s)

        if mse_ep < best:
            best = mse_ep
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    print("Done.")

if __name__ == "__main__":
    main()