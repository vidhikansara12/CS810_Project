import torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalVAE(nn.Module):
    """
    Simple conditional VAE for MNIST.
    Condition via learned label embedding concatenated to input / latent.
    """
    def __init__(self, z_dim=20, hidden_dim=256, num_classes=10, img_dim=28*28, y_embed_dim=32):
        super().__init__()
        self.z_dim = z_dim
        self.img_dim = img_dim
        self.num_classes = num_classes

        self.y_emb = nn.Embedding(num_classes, y_embed_dim)

        enc_in = img_dim + y_embed_dim
        self.enc_fc1 = nn.Linear(enc_in, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, z_dim)
        self.enc_logvar = nn.Linear(hidden_dim, z_dim)

        dec_in = z_dim + y_embed_dim
        self.dec_fc1 = nn.Linear(dec_in, hidden_dim)
        self.dec_out = nn.Linear(hidden_dim, img_dim)  # logits

    def encode(self, x, y):
        # x: (B,1,28,28) in [0,1]
        B = x.size(0)
        x = x.view(B, -1)
        yv = self.y_emb(y)
        h = F.relu(self.enc_fc1(torch.cat([x, yv], dim=1)))
        mu = self.enc_mu(h)
        logvar = self.enc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, y):
        yv = self.y_emb(y)
        h = F.relu(self.dec_fc1(torch.cat([z, yv], dim=1)))
        logits = self.dec_out(h)
        return logits  # (B, 784)

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z, y)
        return logits, mu, logvar