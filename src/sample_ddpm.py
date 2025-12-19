import os, argparse, json
import torch
from torchvision.utils import save_image

from src.models.unet_mnist import TinyConditionalUNet
from src.models.ddpm import DDPM

@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--n_per_class", type=int, default=64)
    p.add_argument("--T", type=int, default=None)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt["args"]
    T = args.T if args.T is not None else cfg["T"]

    model = TinyConditionalUNet(in_ch=1, base_ch=cfg.get("base_ch", 64)).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ddpm = DDPM(T=T, beta_start=cfg.get("beta_start", 1e-4), beta_end=cfg.get("beta_end", 0.02), device=device)

    ys = torch.arange(10, device=device).repeat_interleave(args.n_per_class)
    x = ddpm.sample(model, n=ys.size(0), y=ys, shape=(1,28,28))
    x = (x + 1.0) / 2.0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_image(x, args.out, nrow=args.n_per_class)

    manifest = []
    # NOTE: for now we only save a grid image; later we’ll save individual samples if needed for metrics
    for i in range(ys.size(0)):
        manifest.append({"index": i, "label_intended": int(ys[i].item())})
    mpath = os.path.splitext(args.out)[0] + "_manifest.json"
    with open(mpath, "w") as f:
        json.dump({"grid_path": args.out, "items": manifest}, f, indent=2)

    print("Saved:", args.out)
    print("Manifest:", mpath)

if __name__ == "__main__":
    main()