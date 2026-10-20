"""Building blocks of OccJEPA (verbatim from the research code that produced the paper's numbers).

  MaskHead / HiResHead   the mask head H: token features + visible masks -> amodal logits (128, refined to 256)
  add_lora               LoRA (rank 16) on every linear layer of the last encoder blocks
  drop_sets              occlusion as masking: keep the tokens covered by the visible mask (coverage >= tau)
  fill_tokens            latent completion: the predictor fills the dropped positions, observed tokens elsewhere
  make_transformer       copy of the released predictor ('pre') or the same architecture randomly initialised ('scr')
  token_cover            per-token visible-mask coverage rho_i
"""
from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vjepa as V


class MaskHead(nn.Module):
    """Token features [B, T', h, w, Din] + modal masks [B, T, 128, 128] -> amodal logits [B, T, 128, 128]."""

    def __init__(self, din: int, d: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(din)
        self.proj = nn.Linear(din, d)

        def blk(ci, co, up):
            layers = [nn.ConvTranspose2d(ci, co, 2, 2)] if up else [nn.Conv2d(ci, co, 3, padding=1)]
            return layers + [nn.GroupNorm(8, co), nn.GELU()]

        self.up1 = nn.Sequential(*blk(d, 128, True), *blk(128, 128, False))      # 32
        self.up2 = nn.Sequential(*blk(128, 64, True))                             # 64
        self.fuse2 = nn.Sequential(*blk(64 + 2, 64, False))
        self.up3 = nn.Sequential(*blk(64, 32, True))                              # 128
        self.fuse3 = nn.Sequential(*blk(32 + 2, 32, False), nn.Conv2d(32, 2, 1))

    def forward(self, x, modal128):
        B, Tp, h, w, _ = x.shape
        z = F.gelu(self.proj(self.norm(x))).permute(0, 1, 4, 2, 3).reshape(B * Tp, -1, h, w)
        z = self.up2(self.up1(z))
        m = modal128.reshape(B * Tp, 2, *modal128.shape[-2:])
        z = self.fuse2(torch.cat([z, F.avg_pool2d(m, 2)], 1))
        z = self.fuse3(torch.cat([self.up3(z), m], 1))
        return z.reshape(B, Tp * 2, *z.shape[-2:])


def official_init_(pred: nn.Module, init_std: float = 0.02):
    """Random init exactly like VisionTransformerPredictor.__init__ (_init_weights + _rescale_blocks; mask tokens
    zero-init as zero_init_mask_tokens=True)."""
    for m in pred.modules():
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    with torch.no_grad():
        for i, blk in enumerate(pred.predictor_blocks):
            blk.attn.proj.weight.div_(math.sqrt(2.0 * (i + 1)))
            blk.mlp.fc2.weight.div_(math.sqrt(2.0 * (i + 1)))
        if pred.mask_tokens is not None:
            for mt in pred.mask_tokens:
                mt.zero_()
        if getattr(pred, "modality_embedding", False):
            nn.init.normal_(pred.video_mod_embed, std=1e-6)
            nn.init.normal_(pred.img_mod_embed, std=1e-6)
    return pred


def make_transformer(released: nn.Module, init: str, in_dim: int = None, out_dim: int = None):
    """Copy of the released predictor (init='pre') or the same architecture with the official random init
    ('scr').  in_dim/out_dim replace the input embedding / context projection (for DINOv3 tokens)."""
    p = copy.deepcopy(released).float()
    if init == "scr":
        official_init_(p)
    if in_dim is not None:
        e = p.predictor_embed
        hid = e[0].out_features
        p.predictor_embed = nn.Sequential(nn.Linear(in_dim, hid), nn.GELU(), nn.Linear(hid, e[2].out_features))
        for m in p.predictor_embed.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
    if out_dim is not None:
        pc = p.predictor_proj_context
        p.predictor_proj_context = nn.Linear(pc.in_features, out_dim)
        nn.init.trunc_normal_(p.predictor_proj_context.weight, std=0.02)
        nn.init.zeros_(p.predictor_proj_context.bias)
    for q in p.parameters():
        q.requires_grad_(True)
    p.train()
    return p


def context_path(T: nn.Module, x: torch.Tensor, grid):
    """Transformer on ALL tokens through the predictor's context path (no mask tokens): [B,N,Din] -> [B,N,Dout]."""
    B, N, _ = x.shape
    idx = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
    empty = idx[:, :0]
    _, ctx = V.predictor_forward(T, x, idx, empty, grid)
    return ctx


def fill_tokens(P: nn.Module, ctx_kept: torch.Tensor, kept: torch.Tensor, drop: torch.Tensor, tgt_full: torch.Tensor,
                grid):
    """One sample.  ctx_kept [1,Nk,D] = encoder(clip, masks=kept); returns fill [N, Dt]: predictor output on
    ``drop``, tgt_full (observed) elsewhere."""
    pred, _ = V.predictor_forward(P, ctx_kept, kept[None], drop[None], grid)
    out = tgt_full.clone().to(pred.dtype)
    out = out.index_copy(0, drop, pred[0])
    return out


def drop_sets(cov_target: torch.Tensor, cov_occl: torch.Tensor, kind: str, thr: float = 0.25):
    """Per-sample (kept, drop) index tensors from token coverages [N] of the target's visible pixels / occluders."""
    if kind == "b1":
        drop = cov_occl >= thr
    elif kind == "b2":
        drop = cov_target < thr
    else:
        raise ValueError(kind)
    if drop.all():  # always keep at least one token (the most-visible one)
        drop = drop.clone()
        drop[torch.argmax(cov_target)] = False
    if not drop.any():
        drop = drop.clone()
        drop[torch.argmin(cov_target)] = True
    kept = torch.nonzero(~drop).flatten()
    dr = torch.nonzero(drop).flatten()
    return kept, dr


class LoRALinear(nn.Module):
    """y = base(x) + scale * B(A(x)); A ~ kaiming-uniform, B = 0 (identity at init).  Base stays frozen."""

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 16.0):
        super().__init__()
        self.base = base
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scale = alpha / r

    def forward(self, x):
        return self.base(x) + (x @ self.lora_A.t().to(x.dtype) @ self.lora_B.t().to(x.dtype)) * self.scale


def add_lora(encoder: nn.Module, first_block: int, r: int = 16):
    """Wrap every nn.Linear inside encoder.blocks[first_block:] with LoRA; returns the LoRA parameters."""
    params = []
    for blk in encoder.blocks[first_block:]:
        for name, mod in list(blk.named_modules()):
            for cname, child in list(mod.named_children()):
                if isinstance(child, nn.Linear):
                    lo = LoRALinear(child, r).to(child.weight.device)
                    setattr(mod, cname, lo)
                    params += [lo.lora_A, lo.lora_B]
    return params


HR_DEFAULT_CFG = dict(refine=1, rgb=1, out_res=256, c_stem=16, c_body=48, ref_seed=1234, version=2)


class Cue:
    """High-resolution cues of a batch; ``cue[s]`` slices the batch dimension of every member (like a tensor)."""

    __slots__ = ("m128", "m256", "rgb")

    def __init__(self, m128, m256, rgb):
        self.m128, self.m256, self.rgb = m128, m256, rgb   # [B,T,o,o], [B,T,S,S], [B,3,T,S,S]

    def __getitem__(self, s):
        return Cue(self.m128[s], self.m256[s], self.rgb[s])

    @property
    def shape(self):
        return self.m128.shape


def _cna(ci, co, k=3, stride=1, dil=1, groups=8):
    return [nn.Conv2d(ci, co, k, stride=stride, padding=dil * (k // 2), dilation=dil), nn.GroupNorm(groups, co),
            nn.GELU()]


class Refiner(nn.Module):
    """Sub-pixel refiner: every conv runs at 128x128, the 256x256 inputs/outputs go through pixel (un)shuffle.
    Per frame: [RGB 3, modal256 1, sigmoid(up(base))-0.5 1] at 256 -> pixel_unshuffle(2) -> 20 ch at 128 -> stem
    conv (16).  Per tubelet (2 frames stacked in channels): [stem 2x16, base decoder feature 32, base prob 2,
    modal128 2] -> 3 dilated convs (1, 2, 4) -> [body, stem] -> conv to 2 frames x 4 sub-pixels (ZERO-init) ->
    pixel_shuffle(2) -> delta logits at 256.  Output = bilinear up(base logits) + delta."""

    def __init__(self, c_feat=32, c_stem=16, c_body=48):
        super().__init__()
        self.c_stem = c_stem
        self.stem = nn.Sequential(*_cna(5 * 4, c_stem, groups=4))
        self.body = nn.Sequential(*_cna(2 * c_stem + c_feat + 4, c_body), *_cna(c_body, c_body, dil=2),
                                  *_cna(c_body, c_body, dil=4))
        self.out = nn.Conv2d(c_body + 2 * c_stem, 2 * 4, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, base, feat, m128, m256, rgb):
        """base [B,T,o,o] logits, feat [B*T/2, Cf, o, o], m128 [B,T,o,o], m256 [B,T,S,S], rgb [B,T,3,S,S] or None.
        -> [B,T,S,S] float32 logits."""
        B, T, o = base.shape[0], base.shape[1], base.shape[-1]
        S = m256.shape[-1]
        assert S == 2 * o and T % 2 == 0, (S, o, T)
        Tp = T // 2
        up = F.interpolate(base.float().reshape(B * T, 1, o, o), size=(S, S), mode="bilinear", align_corners=False)
        m = m256.reshape(B * T, 1, S, S).float()
        x = rgb.reshape(B * T, 3, S, S).float() if rgb is not None else m.new_zeros(B * T, 3, S, S)
        hi = F.pixel_unshuffle(torch.cat([x, m, torch.sigmoid(up) - 0.5], 1), 2)      # [BT, 20, o, o]
        s = self.stem(hi).reshape(B * Tp, 2 * self.c_stem, o, o)                       # tubelet: 2 frames stacked
        pb = (torch.sigmoid(base.float()) - 0.5).reshape(B * Tp, 2, o, o)
        z = self.body(torch.cat([s, feat, pb, m128.reshape(B * Tp, 2, o, o).float()], 1))
        d = self.out(torch.cat([z, s], 1)).reshape(B * T, 4, o, o)
        delta = F.pixel_shuffle(d, 2)                                                   # [BT, 1, S, S]
        return (up + delta.float()).reshape(B, T, S, S)


class HiResHead(MaskHead):
    """MaskHead (identical parameters, init and plain-tensor forward) + Refiner (forked RNG at init)."""

    def __init__(self, din: int, d: int = 256, cfg: dict | None = None):
        super().__init__(din, d)
        cfg = dict(HR_DEFAULT_CFG, **(cfg or {}))
        self.refine, self.use_rgb = bool(cfg["refine"]), bool(cfg["rgb"])
        with torch.random.fork_rng(devices=[]):   # keep the global torch RNG stream unchanged
            torch.manual_seed(int(cfg["ref_seed"]))
            self.ref = Refiner(32, cfg["c_stem"], cfg["c_body"])

    def base(self, x, modal128):
        """= MaskHead.forward, op by op, also returning the last 32-channel decoder feature."""
        B, Tp, h, w, _ = x.shape
        z = F.gelu(self.proj(self.norm(x))).permute(0, 1, 4, 2, 3).reshape(B * Tp, -1, h, w)
        z = self.up2(self.up1(z))
        m = modal128.reshape(B * Tp, 2, *modal128.shape[-2:])
        z = self.fuse2(torch.cat([z, F.avg_pool2d(m, 2)], 1))
        f = self.fuse3[:-1](torch.cat([self.up3(z), m], 1))
        lg = self.fuse3[-1](f)
        return lg.reshape(B, Tp * 2, *lg.shape[-2:]), f

    def forward(self, x, cue):
        if isinstance(cue, torch.Tensor):     # plain modal128: the base head only
            return MaskHead.forward(self, x, cue)
        lg, f = self.base(x, cue.m128)
        if not self.refine:
            return lg
        rgb = cue.rgb.permute(0, 2, 1, 3, 4) if self.use_rgb else None      # [B,3,T,S,S] -> [B,T,3,S,S]
        return self.ref(lg, f, cue.m128, cue.m256, rgb)


def token_cover(mask, tub=2, patch=16):
    """mask [B,T,S,S] float -> per-token coverage [B, T/tub * S/patch * S/patch] in V-JEPA (t,h,w) order."""
    c = F.avg_pool3d(mask[:, None], kernel_size=(tub, patch, patch))[:, 0]
    return c.flatten(1)


def seg_loss(logits, target):
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((-1, -2))
    den = p.sum((-1, -2)) + target.sum((-1, -2))
    dice = 1 - (2 * inter + 1) / (den + 1)
    return bce + dice.mean()
