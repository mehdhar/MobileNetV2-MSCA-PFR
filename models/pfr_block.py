import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_


class GroupNorm(nn.GroupNorm):
    """GroupNorm with 1 group (equivalent to LayerNorm across channels)."""

    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)


class PoolAttn(nn.Module):
    """
    Pooling-based Attention (PoolAttn)
    (Unmodified behavior — exactly as provided)
    """

    def __init__(self, dim=256, norm_layer=GroupNorm):
        super().__init__()

        # Pooling operations for spatial and channel mixing
        self.patch_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.patch_pool2 = nn.AdaptiveAvgPool2d((4, None))

        self.embdim_pool1 = nn.AdaptiveAvgPool2d((None, 4))
        self.embdim_pool2 = nn.AdaptiveAvgPool2d((4, None))

        self.norm = norm_layer(dim)

        # Depthwise conv projections
        self.proj0 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.proj1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)
        self.proj2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # Spatial pooling attention
        x_patch1 = self.patch_pool1(x)
        x_patch2 = self.patch_pool2(x)
        x_patch_attn = x_patch1 @ x_patch2
        x_patch_attn = self.proj0(x_patch_attn)

        # Channel mixing via reshaping + pooling
        x_reshaped = x.view(B, C, H * W).transpose(1, 2).view(B, H * W, 32, -1)
        x_emb1 = self.embdim_pool1(x_reshaped)
        x_emb2 = self.embdim_pool2(x_reshaped)

        x_emb = x_emb1 @ x_emb2
        x_emb = x_emb.view(B, H * W, C).transpose(1, 2).view(B, C, H, W)
        x_emb = self.proj1(x_emb)

        # Combine & refine
        out = self.norm(x_patch_attn + x_emb)
        out = self.proj2(out)

        return out


class Mlp(nn.Module):
    """MLP block used inside PoolFormerBlock."""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class PoolFormerBlock(nn.Module):
    """
    PFR Block (Pool-based Feature Refinement)
    (Unmodified, full fidelity to your original code)
    """

    def __init__(
        self,
        dim,
        pool_size=3,
        mlp_ratio=4.,
        act_layer=nn.GELU,
        norm_layer=GroupNorm,
        drop=0.,
        drop_path=0.,
        use_layer_scale=True,
        layer_scale_init_value=1e-5
    ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.token_mixer = PoolAttn(dim=dim, norm_layer=norm_layer)

        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop
        )

        self.drop_path = nn.Identity()  # DropPath intentionally unchanged
        self.use_layer_scale = use_layer_scale

        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value * torch.ones(dim))
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value * torch.ones(dim))

    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(
                self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) *
                self.token_mixer(self.norm1(x))
            )
            x = x + self.drop_path(
                self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) *
                self.mlp(self.norm2(x))
            )
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
