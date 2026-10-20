"""OccJEPA: occlusion as masking with the released V-JEPA 2.1 predictor.

One object, one clip of T=24 frames cropped to S x S around the object's visible mask (see crops.py):
  1. coverage rho_i of each tubelet token by the visible mask; keep K = {rho_i >= tau}, drop D = the rest (drop_sets);
  2. the frozen encoder with LoRA (shared) reads all tokens -> target features z (per-level layer norm), and the kept
     tokens only -> context E(x_K); the pre-trained predictor fills the dropped positions -> f (fill_tokens);
  3. the head H decodes [z; f] with the visible masks and the RGB crop into amodal logits (128, refined to 256).

Variants ("arms", as in the paper's ablations): B2_lora_ft (OccJEPA), B2_lora_scr (the same predictor randomly
initialised), C_lora (no predictor). Initialisation order and seeds are those of the runs reported in the paper, so a
released checkpoint reproduces the paper's predictions. Checkpoints: {arm: {head, T, lora}}.
"""
from __future__ import annotations

import copy
import math

import torch

from . import vjepa as V
from .layers import HR_DEFAULT_CFG, Cue, HiResHead, add_lora, drop_sets, fill_tokens, make_transformer, seg_loss, token_cover

ARMS = {
    "B2_lora_ft": dict(kind="lfill", drop="b2", init="pre"),     # OccJEPA
    "B2_lora_scr": dict(kind="lfill", drop="b2", init="scr"),    # same predictor architecture, random initialisation
    "C_lora": dict(kind="lora", init="pre"),                     # no predictor
}
DEFAULTS = dict(S=256, T=24, thr=0.25, lora_first=36, lora_r=16, lr_head=1e-3, lr_pre=1e-4, lr_scr=1e-3, lr_lora=5e-4,
                wd=0.05, warmup=300, steps=6000)


class OccJEPA:
    def __init__(self, arm="B2_lora_ft", world_model="vitG", hr_cfg=None, device="cuda", ckpt_dir=None, **kw):
        cfg = dict(DEFAULTS, **kw)
        self.cfg, self.arm, self.world_model = cfg, arm, world_model
        assert V.MODEL_ZOO[world_model]["teacher_embed_dim"] is None, \
            "the release supports the 4-level world models vitg / vitG (the paper's distilled ViT-B / ViT-L need the " \
            "teacher-space head of the scaling study)"
        self.n_lv = V.MODEL_ZOO[world_model]["n_levels"]
        S, dev = cfg["S"], device
        self.m = V.load_vjepa21(world_model, ckpt_dir=ckpt_dir or V.DEFAULT_CKPT_DIR, img_size=(S, S), num_frames=24,
                                device=dev, dtype=torch.bfloat16)
        self.grid = self.m.grid(cfg["T"], (S, S))
        self.Dt = self.m.tgt_dim
        released = self.m.predictor
        c = dict(ARMS[arm])
        scr_base = None
        if c.get("init") == "scr":           # one official random init (seed 0), as in the paper's runs
            torch.manual_seed(0)
            scr_base = make_transformer(released, "scr").cpu()
        T = None
        if c["kind"] == "lfill":
            T = make_transformer(released, "pre") if c["init"] == "pre" else make_transformer(scr_base, "pre")
            T = T.to(dev)
        enc = copy.deepcopy(self.m.encoder)
        torch.manual_seed(3)
        lora_params = add_lora(enc, cfg["lora_first"], cfg["lora_r"])
        enc.eval()
        din = 2 * self.Dt if c["kind"] == "lfill" else self.Dt
        self.hr_cfg = dict(HR_DEFAULT_CFG, **(hr_cfg or {}))
        torch.manual_seed(2)
        head = HiResHead(din, cfg=self.hr_cfg).to(dev)
        self.enc, self.T, self.head, self.kind, self.drop = enc, T, head, c["kind"], c.get("drop")
        self.init = c["init"]
        self.lora_params = lora_params
        self.opt = None

    # ------------------------------------------------------------------ parameters / checkpoints
    def param_groups(self):
        cfg = self.cfg
        g = [dict(params=self.head.parameters(), lr=cfg["lr_head"], weight_decay=cfg["wd"])]
        if self.T is not None:
            lr_t = cfg["lr_pre"] if self.init == "pre" else cfg["lr_scr"]
            dec = [p for k, p in self.T.named_parameters() if p.ndim > 1 and "mask_tokens" not in k]
            nod = [p for k, p in self.T.named_parameters() if not (p.ndim > 1 and "mask_tokens" not in k)]
            g += [dict(params=dec, lr=lr_t, weight_decay=cfg["wd"]), dict(params=nod, lr=lr_t, weight_decay=0.0)]
        g += [dict(params=self.lora_params, lr=cfg["lr_lora"], weight_decay=0.0)]
        for x in g:
            x["base_lr"] = x["lr"]
        return g

    def make_optimizer(self):
        self.opt = torch.optim.AdamW(self.param_groups(), betas=(0.9, 0.999))
        return self.opt

    def state_dict(self):
        return {self.arm: dict(head=self.head.state_dict(), T=self.T.state_dict() if self.T is not None else None,
                               lora={k: v for k, v in self.enc.state_dict().items() if "lora_" in k})}

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        ck = torch.load(path, map_location="cuda", weights_only=True)[self.arm]
        self.head.load_state_dict(ck["head"])
        if self.T is not None:
            self.T.load_state_dict(ck["T"])
        missing = self.enc.load_state_dict(ck["lora"], strict=False)
        assert not missing.unexpected_keys, missing.unexpected_keys
        self.eval()
        return self

    def eval(self):
        self.head.eval()
        if self.T is not None:
            self.T.eval()
        return self

    def train(self):
        self.head.train()
        if self.T is not None:
            self.T.train()
        return self

    # ------------------------------------------------------------------ forward
    def tok(self, f):  # [B,N,D] -> [B,T',h,w,D]
        g = self.grid
        return f.reshape(f.shape[0], g.t, g.h, g.w, -1)

    @torch.no_grad()
    def encode(self, b):
        """batch b: clip [B,3,T,S,S] uint8-range, modal [B,T,S,S], occl [B,T,S,S] (zeros), modal_t [B,T,o,o]."""
        x = V.normalize_clip(b["clip"])
        covT, covO = token_cover(b["modal"]), token_cover(b["occl"])
        sets = [drop_sets(covT[i], covO[i], self.drop, self.cfg["thr"]) for i in range(x.shape[0])] \
            if self.kind == "lfill" else None
        return dict(x=x, sets=sets, modal256=b["modal"])

    def logits(self, e, modal_t, i=None):
        """amodal logits [B (or 1), T, 256, 256] in crop space; lfill arms run one sample at a time (i)."""
        cue = modal_t if isinstance(modal_t, Cue) else Cue(modal_t, e["modal256"], e["x"])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if self.kind == "lora":
                h = self.enc(e["x"].to(torch.bfloat16))
                feats = V.per_level_layer_norm(h.float(), self.n_lv)
            else:
                kept, dr = e["sets"][i]
                xi = e["x"][i:i + 1].to(torch.bfloat16)
                tg = V.per_level_layer_norm(self.enc(xi).float(), self.n_lv)[0]
                ck = self.enc(xi, masks=kept[None])
                fill = fill_tokens(self.T, ck, kept, dr, tg, self.grid).float()
                feats = torch.cat([tg, fill], -1)[None]
                cue = cue[i:i + 1]
            return self.head(self.tok(feats), cue)

    @torch.no_grad()
    def predict(self, b):
        e = self.encode(b)
        n = b["clip"].shape[0]
        if self.kind == "lfill":
            return torch.cat([self.logits(e, b["modal_t"], i) for i in range(n)])
        return self.logits(e, b["modal_t"])

    # ------------------------------------------------------------------ training
    def train_step(self, b, step):
        cfg = self.cfg
        e = self.encode(b)
        mt, tgt = b["modal_t"], b["amodal_t"]
        B = mt.shape[0]
        w, n = cfg["warmup"], cfg["steps"]
        lr_mult = (step + 1) / w if step < w else 0.5 * (1 + math.cos(math.pi * (step - w) / max(1, n - w)))
        for g in self.opt.param_groups:
            g["lr"] = g["base_lr"] * lr_mult
        self.opt.zero_grad(set_to_none=True)
        if self.kind == "lfill":
            tot = 0.0
            for i in range(B):
                loss = seg_loss(self.logits(e, mt, i), tgt[i:i + 1]) / B
                loss.backward()
                tot += loss.item()
        else:
            loss = seg_loss(self.logits(e, mt), tgt)
            loss.backward()
            tot = loss.item()
        ps = [p for g in self.opt.param_groups for p in g["params"]]
        torch.nn.utils.clip_grad_norm_(ps, 1.0)
        self.opt.step()
        return tot
