import os, csv, argparse, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm

from src.models.vae import ConditionalVAE

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def kl_div(mu, logvar):
    # sum over latent dims, mean over batch
    return (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)).mean()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    p.add_argument("--z_dim", type=int, default=20)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--run_name", type=str, default="vae_beta1")

    p.add_argument("--out_root", type=str, default="/content/drive/MyDrive/CS810")
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    # Outputs
    ckpt_path = os.path.join(args.out_root, f"checkpoints/{args.run_name}.pt")
    log_csv   = os.path.join(args.out_root, f"results/logs/{args.run_name}_train.csv")
    sample_dir = os.path.join(args.out_root, "results/samples")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    tfm = transforms.ToTensor()  # keep MNIST in [0,1] for BCE
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=256, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = ConditionalVAE(z_dim=args.z_dim, hidden_dim=args.hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "recon_bce", "kl"])

    best_loss = 1e9

    def save_samples(epoch):
        model.eval()
        with torch.no_grad():
            # 10 classes, 16 each => 160 images grid
            ys = torch.arange(10, device=device).repeat_interleave(16)
            z = torch.randn(ys.size(0), args.z_dim, device=device)
            logits = model.decode(z, ys)
            x = torch.sigmoid(logits).view(-1, 1, 28, 28)
            out = os.path.join(sample_dir, f"{args.run_name}_samples_ep{epoch:03d}.png")
            save_image(x, out, nrow=16)
        return out

    def save_recons(epoch):
        model.eval()
        with torch.no_grad():
            x, y = next(iter(test_loader))
            x, y = x.to(device), y.to(device)
            logits, mu, logvar = model(x, y)
            xhat = torch.sigmoid(logits).view(-1, 1, 28, 28)
            # stack originals and recon (first 32)
            pair = torch.cat([x[:32], xhat[:32]], dim=0)
            out = os.path.join(sample_dir, f"{args.run_name}_recon_ep{epoch:03d}.png")
            save_image(pair, out, nrow=16)
        return out

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss, running_rec, running_kl = 0.0, 0.0, 0.0
        n = 0
        t0 = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)

            logits, mu, logvar = model(x, y)

            # BCEWithLogits is stable; mean over batch
            x_flat = x.view(x.size(0), -1)
            recon = F.binary_cross_entropy_with_logits(logits, x_flat, reduction="none").sum(dim=1).mean()
            kl = kl_div(mu, logvar)
            loss = recon + args.beta * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            bs = x.size(0)
            running_loss += loss.item() * bs
            running_rec  += recon.item() * bs
            running_kl   += kl.item() * bs
            n += bs

        loss_ep = running_loss / n
        rec_ep  = running_rec / n
        kl_ep   = running_kl / n
        dt = time.time() - t0

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, loss_ep, rec_ep, kl_ep])

        print(f"Epoch {epoch}: loss={loss_ep:.2f} recon={rec_ep:.2f} kl={kl_ep:.2f} time={dt:.1f}s")

        # Save visuals every 5 epochs + final
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            s1 = save_samples(epoch)
            s2 = save_recons(epoch)
            print("Saved:", s1)
            print("Saved:", s2)

        # Save best checkpoint by loss
        if loss_ep < best_loss:
            best_loss = loss_ep
            torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
            print("Saved checkpoint:", ckpt_path)

    print("Done.")

if __name__ == "__main__":
    main()