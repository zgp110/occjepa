"""V-JEPA 2.1 model interface: frozen encoder + released self-supervised predictor (built with the official code of
github.com/facebookresearch/vjepa2 @ 204698b, imported from $VJEPA2_REPO; the released checkpoints live in
$OCCJEPA_WEIGHTS, see scripts/download_weights.py).

Model zoo (names accepted by :func:`load_vjepa21`)::

    name       file                              encoder                         predictor
    vitb_dist  vjepa2_1_vitb_dist_vitG_384.pt    vit_base     (768, 12)          12 blk, out 1664 (ViT-G teacher, 1 level)
    vitl_dist  vjepa2_1_vitl_dist_vitG_384.pt    vit_large   (1024, 24)          12 blk, out 1664 (ViT-G teacher, 1 level)
    vitg       vjepa2_1_vitg_384.pt              vit_giant_xformers (1408, 40)   24 blk, in/out 4x1408 (4 levels)
    vitG       vjepa2_1_vitG_384.pt              vit_gigantic_xformers (1664, 48) 24 blk, in/out 4x1664 (4 levels)

Context space (predictor input): the encoder's 4 hierarchical levels, each LayerNorm'ed by its own norms_block[i].
Target space: the same 4-level output with a parameter-free layer norm per level (V-JEPA 2.1 pre-training targets).
Tokens are flattened in (t', h', w') order with tubelet 2 and patch 16; input frames use ImageNet normalisation.
"""
from __future__ import annotations

import contextlib
import math
import os
import re
import sys
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VJEPA2_REPO = os.environ.get("VJEPA2_REPO", os.path.join(_ROOT, "third_party", "vjepa2"))
DEFAULT_CKPT_DIR = os.environ.get("OCCJEPA_WEIGHTS", os.path.join(_ROOT, "weights"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
PATCH_SIZE = 16
TUBELET_SIZE = 2
TEACHER_EMBED_DIM = 1664  # backbones.py:208 (ViT-G embed dim)

# Hub constructor arguments, copied from src/hub/backbones.py:288-339.
MODEL_ZOO = {
    "vitb_dist": dict(
        hub_name="vjepa2_1_vit_base_384",
        file="vjepa2_1_vitb_dist_vitG_384",
        checkpoint_key="ema_encoder",
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=TEACHER_EMBED_DIM,
        embed_dim=768,
        n_levels=1,
    ),
    "vitl_dist": dict(
        hub_name="vjepa2_1_vit_large_384",
        file="vjepa2_1_vitl_dist_vitG_384",
        checkpoint_key="ema_encoder",
        predictor_depth=12,
        predictor_num_mask_tokens=8,
        n_output_distillation=1,
        return_all_tokens=True,
        teacher_embed_dim=TEACHER_EMBED_DIM,
        embed_dim=1024,
        n_levels=1,
    ),
    "vitg": dict(
        hub_name="vjepa2_1_vit_giant_384",
        file="vjepa2_1_vitg_384",
        checkpoint_key="target_encoder",
        predictor_depth=24,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        teacher_embed_dim=None,
        embed_dim=1408,
        n_levels=4,
    ),
    "vitG": dict(
        hub_name="vjepa2_1_vit_gigantic_384",
        file="vjepa2_1_vitG_384",
        checkpoint_key="target_encoder",
        predictor_depth=24,
        predictor_num_mask_tokens=8,
        n_output_distillation=4,
        return_all_tokens=True,
        teacher_embed_dim=None,
        embed_dim=1664,
        n_levels=4,
    ),
}

# Only mask token 0 is trained in all released 2.1 predictors we inspected
# (tokens 1..7 are exactly zero in vitl_dist / vitb_dist; see docs).  The
# official predictor.forward default is mask_index=1 (predictor.py:214) and the
# EK100 anticipation wrapper does not override it
# (evals/action_anticipation_frozen/modelcustom/vit_encoder_predictor_concat_ar.py:212).
DEFAULT_MASK_INDEX = 0


# --------------------------------------------------------------------------- #
#  Official-code import
# --------------------------------------------------------------------------- #
def _import_official():
    if VJEPA2_REPO not in sys.path:
        sys.path.insert(0, VJEPA2_REPO)
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)  # timm.models.layers deprecation
        from src.hub import backbones as hub_backbones  # noqa: F401
        import app.vjepa_2_1.models.predictor as vit_pred  # noqa: F401
        import app.vjepa_2_1.models.vision_transformer as vit_enc  # noqa: F401
    return hub_backbones


# Server Content-Length of each file (curl -sI https://dl.fbaipublicfiles.com/vjepa2/<file>.pt,
# checked 2026-10-15).  Used as an integrity check: results/ is a case-INSENSITIVE CIFS
# mount, so vjepa2_1_vitg_384.pt and vjepa2_1_vitG_384.pt are the SAME path there; vitG
# therefore lives in the sub-directory pretrained/vitG/.
EXPECTED_BYTES = {
    "vitb_dist": 1664223428,
    "vitl_dist": 5151198524,
    "vitg": 16878318788,
    "vitG": 30238058912,
}


def ckpt_path(name: str, ckpt_dir: str = DEFAULT_CKPT_DIR) -> str:
    """``<ckpt_dir>/<file>.pt`` (or ``<ckpt_dir>/<name>/<file>.pt`` if only that exists: case-insensitive file systems
    cannot hold vjepa2_1_vitg_384.pt and vjepa2_1_vitG_384.pt side by side)."""
    fn = MODEL_ZOO[name]["file"] + ".pt"
    sub = os.path.join(ckpt_dir, name, fn)
    if os.path.exists(sub) and (name == "vitG" or not os.path.exists(os.path.join(ckpt_dir, fn))):
        return sub
    return os.path.join(ckpt_dir, fn)


def ckpt_is_done(name: str, ckpt_dir: str = DEFAULT_CKPT_DIR) -> bool:
    """Complete = the file exists and its size equals the server Content-Length (EXPECTED_BYTES)."""
    p = ckpt_path(name, ckpt_dir)
    return os.path.exists(p) and os.path.getsize(p) == EXPECTED_BYTES[name]


# --------------------------------------------------------------------------- #
#  Token grid helpers
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TokenGrid:
    """Token grid of a clip: t' tubelets x h' rows x w' cols (flattened t,h,w)."""

    t: int
    h: int
    w: int

    @classmethod
    def from_clip(cls, num_frames: int, height: int, width: int,
                  tubelet: int = TUBELET_SIZE, patch: int = PATCH_SIZE) -> "TokenGrid":
        assert num_frames % tubelet == 0 and height % patch == 0 and width % patch == 0, (
            f"clip {num_frames}x{height}x{width} not divisible by tubelet {tubelet} / patch {patch}")
        return cls(num_frames // tubelet, height // patch, width // patch)

    @property
    def hw(self) -> int:
        return self.h * self.w

    @property
    def n(self) -> int:
        return self.t * self.h * self.w

    def index(self, t, h, w):
        """Flattened index of token (t', h', w') (ints or tensors)."""
        return t * self.hw + h * self.w + w

    def unravel(self, idx):
        t = idx // self.hw
        r = idx - t * self.hw
        return t, r // self.w, r % self.w

    def tubelet_indices(self, t0: int, t1: int, device=None) -> torch.Tensor:
        """Indices of all tokens with t0 <= t' < t1 (t1 may exceed self.t: future slots)."""
        return torch.arange(t0 * self.hw, t1 * self.hw, device=device, dtype=torch.long)


def grid_shape(num_frames: int, height: int, width: int) -> Tuple[int, int, int]:
    g = TokenGrid.from_clip(num_frames, height, width)
    return g.t, g.h, g.w


def flat_index(t, h, w, grid: TokenGrid):
    return grid.index(t, h, w)


def future_mask(grid: TokenGrid, k_ctx: int, n_future: Optional[int] = None,
                batch_size: int = 1, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
    """Context = first ``k_ctx`` tubelets, target = the next ``n_future`` tubelets.

    Returns (masks_x, masks_y) as LongTensors [B, k_ctx*h*w] and [B, n_future*h*w]
    holding flattened (t',h',w') indices -- the format the predictor expects
    (predictor.py:214-257; masks are 'indices to keep', src/masks/utils.py:9-21).
    ``k_ctx + n_future`` may exceed grid.t (positions beyond the clip are fine for
    the predictor: positions are only used through RoPE).
    """
    if n_future is None:
        n_future = grid.t - k_ctx
    assert k_ctx >= 1 and n_future >= 1
    mx = grid.tubelet_indices(0, k_ctx, device).unsqueeze(0).repeat(batch_size, 1)
    my = grid.tubelet_indices(k_ctx, k_ctx + n_future, device).unsqueeze(0).repeat(batch_size, 1)
    return mx, my


# Mask configs used in V-JEPA 2.1 pretraining (configs/train_2_1/*/pretrain-256px-16f.yaml:72-98)
MASK_CFG_SHORT = dict(npred=8, spatial_scale=(0.15, 0.15), aspect_ratio=(0.75, 1.5))
MASK_CFG_LONG = dict(npred=2, spatial_scale=(0.7, 0.7), aspect_ratio=(0.75, 1.5))


def block_masks(grid: TokenGrid, batch_size: int, npred: int = 2,
                spatial_scale=(0.7, 0.7), aspect_ratio=(0.75, 1.5),
                temporal_scale=(1.0, 1.0), seed: int = 0, device=None):
    """Spatial 'tube' block masks exactly as in pretraining.

    Uses the official ``_MaskGenerator`` (src/masks/multiseq_multiblock3d.py:93-269)
    with crop (h'*16, w'*16) and t'*2 frames.  Blocks span all tubelets
    (temporal_scale 1.0).  Returns (masks_x = context/visible, masks_y = masked)
    LongTensors [B, Nx], [B, Ny]; all rows truncated to equal length like the
    official collator.
    """
    _import_official()
    from src.masks.multiseq_multiblock3d import _MaskGenerator

    gen = _MaskGenerator(
        crop_size=(grid.h * PATCH_SIZE, grid.w * PATCH_SIZE),
        num_frames=grid.t * TUBELET_SIZE,
        spatial_patch_size=(PATCH_SIZE, PATCH_SIZE),
        temporal_patch_size=TUBELET_SIZE,
        spatial_pred_mask_scale=tuple(spatial_scale),
        temporal_pred_mask_scale=tuple(temporal_scale),
        aspect_ratio=tuple(aspect_ratio),
        npred=npred,
    )
    with gen._itr_counter.get_lock():
        gen._itr_counter.value = seed - 1  # block size uses its own seeded generator
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)  # block positions use the global CPU RNG
        mx, my = gen(batch_size)
    return mx.to(device), my.to(device)


def normalize_clip(frames_uint8: torch.Tensor) -> torch.Tensor:
    """uint8/float pixels in [0,255], layout [..., C, T, H, W] -> model input (ImageNet norm)."""
    x = frames_uint8.float() / 255.0
    shape = [1] * x.ndim
    c_dim = x.ndim - 4
    shape[c_dim] = 3
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(shape)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(shape)
    return (x - mean) / std


def split_levels(x: torch.Tensor, n_levels: int):
    """Split a [.., n_levels*D] tensor into its per-level chunks (shallow -> deep)."""
    return list(x.chunk(n_levels, dim=-1))


def per_level_layer_norm(x: torch.Tensor, n_levels: int) -> torch.Tensor:
    """Parameter-free LayerNorm per level chunk (train.py:596-610)."""
    d = x.shape[-1] // n_levels
    return torch.cat([F.layer_norm(c, (d,)) for c in x.split(d, dim=-1)], dim=-1)


# --------------------------------------------------------------------------- #
#  Predictor forward (any grid, explicit mask_index)
# --------------------------------------------------------------------------- #
def _iso_separate_positions(attn):
    """Instance override: isotropic RoPE rescaling (both axes by (G-1)/(H'-1))."""
    cls_fn = type(attn).separate_positions

    def fn(ids, H_patches=None, W_patches=None):
        d, h, w = cls_fn(attn, ids, H_patches, W_patches)
        Hp = H_patches if H_patches is not None else attn.grid_size
        s = (attn.pretrained_grid_size - 1) / (Hp - 1)
        return d, h * s, w * s

    return fn


def set_rope_mode(module: nn.Module, mode) -> None:
    """RoPE spatial-position mode for every RoPEAttention inside ``module``.

    'interp' (or True) : official interpolate_rope=True -- h' and w' are EACH rescaled
                         to [0, 15] (modules.py:227-233); anisotropic for non-square grids.
    'none' (or False)  : raw positions 0..H'-1 / 0..W'-1 (official interpolate_rope=False).
    'iso'              : ours, not in the official code -- both axes scaled by 15/(H'-1)
                         (keeps the pixel aspect; w' then exceeds 15 for wide inputs).
    """
    if mode is True:
        mode = "interp"
    elif mode is False:
        mode = "none"
    assert mode in ("interp", "none", "iso"), mode
    for m in module.modules():
        if hasattr(m, "interpolate_rope") and hasattr(m, "pretrained_grid_size"):
            m.interpolate_rope = mode == "interp"
            if mode == "iso":
                m.separate_positions = _iso_separate_positions(m)
            elif "separate_positions" in m.__dict__:
                del m.separate_positions


def set_rope_interpolation(module: nn.Module, flag) -> None:
    """Alias of :func:`set_rope_mode` (True='interp', False='none', or a mode string)."""
    set_rope_mode(module, flag)


def predictor_forward(predictor: nn.Module, x: torch.Tensor, masks_x: torch.Tensor,
                      masks_y: torch.Tensor, grid: TokenGrid,
                      mask_index: int = DEFAULT_MASK_INDEX, mod: str = "video", return_hidden: bool = False):
    """Re-implementation of VisionTransformerPredictor.forward (predictor.py:214-295)
    that (1) passes the real (T', H', W') grid to every block so RoPE positions are
    right for non-square inputs (the official forward calls ``blk(x, mask=masks)``
    without H/W, predictor.py:275, and then assumes a square grid of side
    img_size[0]/16, modules.py:187-197), (2) does not need ``num_patches`` (mask
    tokens are expanded directly instead of repeat+gather, which is identical
    because one shared token is used), and (3) uses gather instead of the python
    loop for the (no-op w.r.t. attention) argsort reordering.

    x: [B, Nx, D_in] context tokens; masks_x [B, Nx], masks_y [B, Ny] Long.
    Returns (pred [B, Ny, D_out], ctx [B, Nx, D_out] or None).
    """
    if masks_x.dim() == 1:
        masks_x = masks_x.unsqueeze(0).expand(x.shape[0], -1)
    if masks_y.dim() == 1:
        masks_y = masks_y.unsqueeze(0).expand(x.shape[0], -1)
    B, Nx, _ = x.shape
    Ny = masks_y.shape[1]
    assert masks_x.shape == (B, Nx), (masks_x.shape, x.shape)
    x = predictor.predictor_embed(x)
    D = x.shape[-1]
    mi = mask_index % predictor.num_mask_tokens
    pred_tokens = predictor.mask_tokens[mi].to(x.dtype).expand(B, Ny, D)
    x = torch.cat([x, pred_tokens], dim=1)
    masks = torch.cat([masks_x, masks_y], dim=1).to(x.device)
    argsort = torch.argsort(masks, dim=1)
    masks = torch.gather(masks, 1, argsort)
    x = torch.gather(x, 1, argsort.unsqueeze(-1).expand(-1, -1, D))
    if predictor.modality_embedding:
        emb = predictor.img_mod_embed if mod == "image" else predictor.video_mod_embed
        x = x + emb.to(x.dtype)
    for blk in predictor.predictor_blocks:
        x, _ = blk(x, mask=masks, T=grid.t, H_patches=grid.h, W_patches=grid.w)
    x = predictor.predictor_norm(x)
    rev = torch.argsort(argsort, dim=1)
    x = torch.gather(x, 1, rev.unsqueeze(-1).expand(-1, -1, D))
    x_pred = predictor.predictor_proj(x[:, Nx:])
    x_ctx = None
    if getattr(predictor, "return_all_tokens", False):
        x_ctx = predictor.predictor_proj_context(x[:, :Nx])
    if return_hidden:  # normed predictor hidden states at the target positions [B, Ny, predictor_embed_dim]
        return x_pred, x_ctx, x[:, Nx:]
    return x_pred, x_ctx


# --------------------------------------------------------------------------- #
#  Model wrapper
# --------------------------------------------------------------------------- #
class VJepa21(nn.Module):
    """Frozen V-JEPA 2.1 encoder(s) + released predictor.

    Attributes
    ----------
    encoder        : hub encoder (dist: 'ema_encoder', g/G: 'target_encoder').
                     Used for predictor input (context) by default.
    online_encoder : optional checkpoint 'encoder' (the online/context encoder
                     the predictor was trained with); load with load_online=True.
    target_encoder : encoder whose tokens define the target space.
                     g/G: same module as ``encoder`` ('target_encoder' key).
                     dist: the ViT-G teacher (vitG checkpoint), only when
                     load_teacher=True, else None.
    predictor      : released predictor.
    """

    def __init__(self, name, encoder, predictor, target_encoder=None, online_encoder=None,
                 img_size=(384, 384), num_frames=16, info=None):
        super().__init__()
        self.name = name
        self.cfg = MODEL_ZOO[name]
        self.encoder = encoder
        self.predictor = predictor
        self.online_encoder = online_encoder
        # avoid registering the same module twice
        self._target_is_encoder = target_encoder is encoder
        self.teacher = None if self._target_is_encoder else target_encoder
        self.img_size = tuple(img_size)
        self.num_frames = num_frames
        # Low-precision weights MUST run under autocast: the official RoPE builds
        # fp32 sin/cos from fp32 positions (modules.py:28-44), turning q/k into fp32
        # while v stays bf16 -> SDPA dtype error without autocast.  Pretraining also
        # ran under autocast(bf16) (train.py:678).
        wdt = next(predictor.parameters()).dtype
        self.amp_dtype = wdt if wdt in (torch.bfloat16, torch.float16) else None
        self.info = info or {}
        self.n_levels_ctx = self.cfg["n_levels"]          # predictor input levels
        self.n_levels_tgt = self.cfg["n_levels"]          # predictor output levels
        self.embed_dim = self.cfg["embed_dim"]
        self.ctx_dim = self.embed_dim * self.n_levels_ctx
        if self.cfg["teacher_embed_dim"] is not None:
            self.tgt_dim = self.cfg["teacher_embed_dim"] // self.cfg["n_output_distillation"] \
                * len(predictor.hierarchical_layers)
        else:
            self.tgt_dim = self.embed_dim * len(predictor.hierarchical_layers)
        # encoders return hierarchical (4-level) output iff the predictor eats 4 levels
        hier = len(predictor.hierarchical_layers) > 1
        for enc in (encoder, online_encoder):
            if enc is not None:
                enc.return_hierarchical = hier
        if self.teacher is not None:
            self.teacher.return_hierarchical = False  # last layer only (paper App. B)

    # -- properties -------------------------------------------------------- #
    @property
    def target_encoder(self):
        return self.encoder if self._target_is_encoder else self.teacher

    def amp(self):
        """Autocast context matching the weight dtype (no-op for fp32 weights)."""
        # NOTE: return a nullcontext (not autocast(enabled=False)) for fp32 weights so
        # that an OUTER autocast set by the caller stays active.
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        dev = "cuda" if self.device.type == "cuda" else "cpu"
        return torch.autocast(dev, dtype=self.amp_dtype)

    @property
    def device(self):
        return next(self.predictor.parameters()).device

    @property
    def dtype(self):
        return next(self.predictor.parameters()).dtype

    def grid(self, num_frames: Optional[int] = None, img_size=None) -> TokenGrid:
        H, W = img_size or self.img_size
        return TokenGrid.from_clip(num_frames or self.num_frames, H, W)

    # -- encoders ----------------------------------------------------------- #
    @torch.no_grad()
    def encode_context(self, clip: torch.Tensor, masks: Optional[torch.Tensor] = None,
                       online: bool = False) -> torch.Tensor:
        """Predictor-input tokens.  clip [B,3,T,H,W] normalised.  ``masks`` [B,N] Long
        = indices of tokens the encoder may see (others are dropped before the first
        block, vision_transformer.py:293-295, exactly like pretraining)."""
        enc = self.online_encoder if online else self.encoder
        assert enc is not None, "online encoder not loaded (load_online=True)"
        clip = clip.to(self.device, self.dtype)
        with self.amp():
            return enc(clip, masks=masks)

    @torch.no_grad()
    def encode_target(self, clip: torch.Tensor, masks: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Target-space tokens [B, N, tgt_dim] (float32), per-level F.layer_norm'ed."""
        tenc = self.target_encoder
        if tenc is None:
            raise RuntimeError(
                f"{self.name}: target space is the ViT-G teacher; load with load_teacher=True "
                "(needs vjepa2_1_vitG_384.pt DONE)")
        clip = clip.to(next(tenc.parameters()).device, next(tenc.parameters()).dtype)
        with self.amp():
            h = tenc(clip, masks=masks)
        h = h.float()
        n_lv = len(self.predictor.hierarchical_layers)
        return per_level_layer_norm(h, n_lv)

    encode = encode_target  # task API name: tokens used as targets

    # -- predictor ---------------------------------------------------------- #
    @torch.no_grad()
    def predict(self, ctx_tokens: torch.Tensor, masks_x: torch.Tensor, masks_y: torch.Tensor,
                grid: Optional[TokenGrid] = None, mask_index: int = DEFAULT_MASK_INDEX,
                return_context: bool = False):
        """Predicted target-space tokens [B, Ny, tgt_dim] (float32, raw predictor output).

        The predictor output is directly the regression target (L1 to the per-level
        layer-normed targets, train.py:629-675, normalize_predictor=false in all
        released configs), so it lives in the same space as :meth:`encode_target`.
        """
        grid = grid or self.grid()
        with self.amp():
            pred, ctx = predictor_forward(self.predictor, ctx_tokens.to(self.dtype), masks_x,
                                          masks_y, grid, mask_index=mask_index)
        if return_context:
            return pred.float(), (ctx.float() if ctx is not None else None)
        return pred.float()

    def predict_grad(self, ctx_tokens, masks_x, masks_y, grid=None, mask_index=DEFAULT_MASK_INDEX):
        """Same as :meth:`predict` but with autograd (for predictor tuning).  Wrap the
        call in ``self.amp()`` (or your own autocast) if the weights are bf16."""
        grid = grid or self.grid()
        return predictor_forward(self.predictor, ctx_tokens, masks_x, masks_y, grid,
                                 mask_index=mask_index)

    @torch.no_grad()
    def official_predict(self, ctx_tokens, masks_x, masks_y, mask_index=1):
        """Call the UNMODIFIED official forward (square grid of side img_size[0]/16 only)."""
        with self.amp():
            return self.predictor(ctx_tokens.to(self.dtype), masks_x, masks_y, mask_index=mask_index)

    def split_target_levels(self, t: torch.Tensor):
        return split_levels(t, len(self.predictor.hierarchical_layers))


# --------------------------------------------------------------------------- #
#  Loading
# --------------------------------------------------------------------------- #
def _build_hub_modules(name: str, num_frames: int):
    """Construct (encoder, predictor) with exactly the hub kwargs (pretrained=False)."""
    hub = _import_official()
    c = MODEL_ZOO[name]
    return hub._make_vjepa2_1_model(
        model_name=c["hub_name"],
        checkpoint_key=c["checkpoint_key"],
        img_size=384,  # only affects predictor.num_patches / square-grid fallback, see docs
        num_frames=num_frames,
        predictor_num_mask_tokens=c["predictor_num_mask_tokens"],
        predictor_depth=c["predictor_depth"],
        n_output_distillation=c["n_output_distillation"],
        return_all_tokens=c["return_all_tokens"],
        teacher_embed_dim=c["teacher_embed_dim"],
        pretrained=False,
    )


def _load_sd(module: nn.Module, sd: dict) -> None:
    hub = _import_official()
    sd = hub._clean_backbone_key(dict(sd))
    module.load_state_dict(sd, strict=True)


def _load_ckpt(name: str, ckpt_dir: str, require_done: bool = True):
    p = ckpt_path(name, ckpt_dir)
    if require_done and not ckpt_is_done(name, ckpt_dir=ckpt_dir):
        raise FileNotFoundError(f"{p}: missing or incomplete (expected {EXPECTED_BYTES[name]} bytes; "
                                f"run scripts/download_weights.py)")
    return torch.load(p, map_location="cpu", mmap=True, weights_only=True)


def load_vjepa21(name: str, ckpt_dir: str = DEFAULT_CKPT_DIR, img_size=(384, 384),
                 num_frames: int = 16, device: Union[str, torch.device] = "cuda",
                 dtype: torch.dtype = torch.float32, load_online: bool = False,
                 load_teacher: bool = False, teacher_key: str = "target_encoder",
                 encoder_rope_interp="interp", predictor_rope_interp="interp",
                 require_done: bool = True) -> VJepa21:
    """Build encoder + predictor exactly like ``src/hub/backbones.py:_make_vjepa2_1_model``
    and load the local checkpoint with strict=True for BOTH (as the hub does,
    backbones.py:276-283).  Returns a :class:`VJepa21` in eval mode, frozen.

    encoder_rope_interp / predictor_rope_interp: RoPE mode ('interp' / 'none' / 'iso',
    or True/False), see :func:`set_rope_mode`.  The hub sets interpolate_rope=True for
    the encoder only (backbones.py:240; predictor default False, predictor.py:54), but
    pretraining passed the config value (``interpolate_rope: true``) to BOTH
    (train.py:109,358; utils.py:232,261).  DEFAULT HERE = 'interp' for both, i.e. the
    training setting, which DEVIATES from the hub predictor; measured on real video it
    predicts clearly better on real video.  Pass predictor_rope_interp=False
    to reproduce the hub exactly.
    load_teacher (dist models only): also load the ViT-G teacher (vitG checkpoint,
    key ``teacher_key``) to compute targets.
    """
    if name not in MODEL_ZOO:
        raise KeyError(f"unknown model {name}; choose from {list(MODEL_ZOO)}")
    c = MODEL_ZOO[name]
    device = torch.device(device)
    sd = _load_ckpt(name, ckpt_dir, require_done)
    info = {"ckpt": ckpt_path(name, ckpt_dir), "ckpt_keys": list(sd.keys()),
            "encoder_key": c["checkpoint_key"]}
    for k in ("epoch", "loss", "batch_size", "world_size", "lr"):
        if k in sd:
            info[k] = sd[k]
    with torch.device(device):
        encoder, predictor = _build_hub_modules(name, num_frames)
    _load_sd(encoder, sd[c["checkpoint_key"]])
    _load_sd(predictor, sd["predictor"])
    online = None
    if load_online:
        with torch.device(device):
            online, _ = _build_hub_modules(name, num_frames)
        _load_sd(online, sd["encoder"])
        info["online_key"] = "encoder"
    del sd

    if c["teacher_embed_dim"] is None:
        target = encoder
        info["target"] = f"{name}:{c['checkpoint_key']} (4-level, per-level F.layer_norm)"
    elif load_teacher:
        tsd = _load_ckpt("vitG", ckpt_dir, require_done)
        with torch.device(device):
            target, _ = _build_hub_modules("vitG", num_frames)
        _load_sd(target, tsd[teacher_key])
        del tsd
        info["target"] = f"vitG:{teacher_key} last layer (norms_block[-1]) + F.layer_norm"
    else:
        target = None
        info["target"] = "ViT-G teacher not loaded (load_teacher=False)"

    mods = [m for m in (encoder, predictor, online, target) if m is not None]
    for m in mods:
        m.to(device=device, dtype=dtype)
        m.eval()
        for p in m.parameters():
            p.requires_grad_(False)
    for m in (encoder, online, target):
        if m is not None:
            set_rope_interpolation(m, encoder_rope_interp)
    set_rope_interpolation(predictor, predictor_rope_interp)
    info["encoder_rope_interp"] = encoder_rope_interp
    info["predictor_rope_interp"] = predictor_rope_interp
    return VJepa21(name, encoder, predictor, target_encoder=target, online_encoder=online,
                   img_size=img_size, num_frames=num_frames, info=info)


__all__ = [
    "MODEL_ZOO", "IMAGENET_MEAN", "IMAGENET_STD", "TokenGrid", "grid_shape", "flat_index",
    "future_mask", "block_masks", "MASK_CFG_SHORT", "MASK_CFG_LONG", "normalize_clip",
    "split_levels", "per_level_layer_norm", "predictor_forward", "set_rope_interpolation",
    "set_rope_mode", "VJepa21", "load_vjepa21", "ckpt_is_done", "ckpt_path",
    "DEFAULT_MASK_INDEX", "EXPECTED_BYTES",
]
