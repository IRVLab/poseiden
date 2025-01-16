import torch
import torch.nn as nn
from einops import rearrange, repeat


# Pos embedding
def pos_emb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature ** omega)

    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        if project_out:
            self.to_out = nn.Linear(inner_dim, dim, bias=False)
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        q = self.to_q(x)
        k = self.to_k(x)
        v = self.to_v(x)

        # Split the embedding into self.heads different pieces
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, head_dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attnlayer, ff in self.layers:
            message, attnmap = attnlayer(x)
            x = message + x
            x = ff(x) + x

        return x, attnmap


class FeatureTransformer(nn.Module):
    def __init__(self, embed_size, feature_size, nheads, nlayers):
        super(FeatureTransformer, self).__init__()

        self.pos_embedding = pos_emb_sincos_2d(
            h=feature_size,
            w=feature_size,
            dim=embed_size,
        )

        self.transformer = Transformer(
            embed_size, nlayers, nheads, embed_size // nheads, embed_size, 0)

    def forward(self, featl, featr=None):
        """
        Args:
            featl (torch.Tensor): [N, C, H, W]
            featr (torch.Tensor): [N, C, H, W]
        """
        b, c, h, w = featl.size()

        if featr is not None:
            x0 = rearrange(featl, 'b c h w -> b (h w) c')
            x1 = rearrange(featr, 'b c h w -> b (h w) c')

            pos0 = self.pos_embedding.to(x0.device)
            x0 += repeat(pos0, 'n d -> b n d', b=b)

            pos1 = self.pos_embedding.to(x1.device)
            x1 += repeat(pos1, 'n d -> b n d', b=b)

            x = torch.cat((x0, x1), dim=1)
            x, attnmap = self.transformer(x)
            x0, x1 = torch.chunk(x, 2, dim=1)

            featl = rearrange(x0, 'b (h w) c -> b c h w', h=h, w=w)
            featr = rearrange(x1, 'b (h w) c -> b c h w', h=h, w=w)

            return featl, featr, attnmap

        else:
            x = rearrange(featl, 'b c h w -> b (h w) c')
            pos = self.pos_embedding.to(x.device)
            x += repeat(pos, 'n d -> b n d', b=b)

            x, attnmap = self.transformer(x)
            featl = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

            return featl, attnmap
