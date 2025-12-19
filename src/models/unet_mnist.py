import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def timestep_embedding(t, dim: int):
    """
    Sinusoidal timestep embedding.
    t: (B,) int64 or float
    returns: (B, dim)
    """
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=t.device).float() / (half - 1))
    args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_dim, y_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.t_proj = nn.Linear(t_dim, out_ch)
        self.y_proj = nn.Linear(y_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb, y_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        # add time + class bias
        h = h + self.t_proj(t_emb).unsqueeze(-1).unsqueeze(-1) + self.y_proj(y_emb).unsqueeze(-1).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 4, stride=2, padding=1)  # /2

    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1)  # *2

    def forward(self, x):
        return self.conv(x)

class TinyConditionalUNet(nn.Module):
    """
    Tiny UNet for MNIST (28x28), predicts noise epsilon.
    Conditional via label embedding + timestep embedding.
    """
    def __init__(self, in_ch=1, base_ch=64, t_dim=128, num_classes=10, y_dim=128):
        super().__init__()
        self.t_dim = t_dim
        self.y_emb = nn.Embedding(num_classes, y_dim)

        self.t_mlp = nn.Sequential(
            nn.Linear(t_dim, t_dim),
            nn.SiLU(),
            nn.Linear(t_dim, t_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down path
        self.rb1 = ResBlock(base_ch, base_ch, t_dim, y_dim)
        self.down1 = Down(base_ch)  # 28->14
        self.rb2 = ResBlock(base_ch, base_ch*2, t_dim, y_dim)
        self.down2 = Down(base_ch*2)  # 14->7

        # Bottleneck
        self.rb3 = ResBlock(base_ch*2, base_ch*2, t_dim, y_dim)

        # Up path
        self.up1 = Up(base_ch*2)  # 7->14
        self.rb4 = ResBlock(base_ch*2 + base_ch*2, base_ch, t_dim, y_dim)
        self.up2 = Up(base_ch)    # 14->28
        self.rb5 = ResBlock(base_ch + base_ch, base_ch, t_dim, y_dim)

        self.out_norm = nn.GroupNorm(8, base_ch)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 3, padding=1)

    def forward(self, x, t, y):
        # embeddings
        t_emb = timestep_embedding(t, self.t_dim)
        t_emb = self.t_mlp(t_emb)
        y_emb = self.y_emb(y)

        x0 = self.in_conv(x)
        h1 = self.rb1(x0, t_emb, y_emb)      # 28
        d1 = self.down1(h1)                  # 14
        h2 = self.rb2(d1, t_emb, y_emb)      # 14
        d2 = self.down2(h2)                  # 7

        mid = self.rb3(d2, t_emb, y_emb)     # 7

        u1 = self.up1(mid)                   # 14
        cat1 = torch.cat([u1, h2], dim=1)
        h3 = self.rb4(cat1, t_emb, y_emb)    # 14

        u2 = self.up2(h3)                    # 28
        cat2 = torch.cat([u2, h1], dim=1)
        h4 = self.rb5(cat2, t_emb, y_emb)    # 28

        out = self.out_conv(F.silu(self.out_norm(h4)))
        return out