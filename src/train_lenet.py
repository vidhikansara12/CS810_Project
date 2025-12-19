import os, csv, argparse, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from src.models.lenet import LeNet

def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def accuracy(logits, y):
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()

@torch.no_grad()
def eval_model(model, loader, device):
    model.eval()
    total_acc, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        bs = x.size(0)
        total_acc += accuracy(logits, y) * bs
        n += bs
    return total_acc / max(n, 1)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=2)

    # Drive output defaults (you can override)
    p.add_argument("--out_root", type=str, default="/content/drive/MyDrive/CS810")
    args = p.parse_args()

    ckpt_path = os.path.join(args.out_root, "checkpoints/lenet.pt")
    log_csv   = os.path.join(args.out_root, "results/logs/lenet_train.csv")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = True

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds  = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=512, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    model = LeNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0

    with open(log_csv, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "test_acc"])

    for epoch in range(1, args.epochs + 1):
        model.train()
        running, n = 0.0, 0
        t0 = time.time()

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            bs = x.size(0)
            running += loss.item() * bs
            n += bs

        train_loss = running / max(n, 1)
        test_acc = eval_model(model, test_loader, device)
        dt = time.time() - t0

        with open(log_csv, "a", newline="") as f:
            csv.writer(f).writerow([epoch, train_loss, test_acc])

        print(f"Epoch {epoch}: train_loss={train_loss:.4f} test_acc={test_acc:.4f} time={dt:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                "model": model.state_dict(),
                "best_test_acc": best_acc,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved best checkpoint: {ckpt_path} (acc={best_acc:.4f})")

    print("Done. Best test acc:", best_acc)

if __name__ == "__main__":
    main()