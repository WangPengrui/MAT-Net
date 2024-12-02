import torch
from torch import nn
from timm.models.layers import DropPath
from timm.models.vision_transformer import Mlp
from functools import partial
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


# helpers
def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class Attention(nn.Module):
    def __init__(self, dim, p, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        dimm = dim[0] * dim[1] * dim[2]
        assert dimm % num_heads == 0, f"Dim should be divisible by heads dim={dim}, heads={num_heads}"
        head_dim = dimm // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.wq = nn.Conv2d(p, p, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=qkv_bias)
        self.wk = nn.Conv2d(p, p, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=qkv_bias)
        self.wv = nn.Conv2d(p, p, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Conv2d(p, p, kernel_size=(1, 1), stride=1, padding=(0, 0), bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, p, c, _ = x.shape
        dim = p * c * c
        q = self.wq(x).reshape(b, self.num_heads, dim // self.num_heads).permute(1, 0, 2)
        k = self.wk(x).reshape(b, self.num_heads, dim // self.num_heads).permute(1, 0, 2)
        v = self.wv(x).reshape(b, self.num_heads, dim // self.num_heads).permute(1, 0, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # BH1(C/H) @ BH(C/H)N -> BH1N
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(0, 1).reshape(b, p, c, c)  # (BH1N @ BHN(C/H)) -> BH1(C/H) -> B1H(C/H) -> B1C
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Mlp(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Conv2d(hidden_dim, dim, kernel_size=(1, 1), stride=1, padding=(0, 0)),
            nn.Dropout(drop)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, dim, p, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0.15, norm_layer=nn.LayerNorm, act_layer=nn.GELU, drop_ratio=0.):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, p, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

        mlp_hidden_dim = int(p * mlp_ratio)
        self.mlp = Mlp(dim=p, hidden_dim=mlp_hidden_dim, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, depth, heads, mlp_dim, qkv_bias=True,qk_scale=None, drop_ratio=0., attn_drop=0.,
                 pool='cls', l, p, dim_head=64, dropout=0., emb_dropout=0., norm_layer=None, act_layer=None):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        # self.pos_embedding = nn.Parameter(torch.randn(1, 5, p, image_size, image_size))
        # self.cls_token = nn.Parameter(torch.randn(1, p, image_size, image_size))
        self.dropout = nn.Dropout(emb_dropout)


        dim = [p, image_size, image_size]
        self.blocks = nn.Sequential(*[
            Block(dim=dim, p=p, mlp_ratio=4, qkv_bias=qkv_bias, qk_scale=qk_scale, num_heads=heads,
                  drop_ratio=drop_ratio, attn_drop=attn_drop, act_layer = act_layer)
            for _ in range(depth)
        ])
        self.norm = norm_layer(dim)


    def forward(self, x):

        # cls_tokens = self.cls_token
        # x = torch.cat((cls_tokens, x), dim=0)
        # x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.blocks(x)
        x = x.mean(dim=0)
        return x


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)