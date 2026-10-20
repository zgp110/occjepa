# OccJEPA: Occlusion as Masking for Amodal Video Segmentation

A video world model is pre-trained to predict what lies in masked regions of a clip. OccJEPA treats the hidden part of
an object as exactly such a region: the tokens that the object's visible mask does not cover are dropped, the
pre-trained V-JEPA 2.1 predictor fills them in, and a light head decodes the observed and the predicted features into
amodal masks. One deterministic model, 33.5x faster than diffusion-based amodal segmentation.

Project page: `docs/` (served with GitHub Pages).

## Results (Diffusion-VAS protocols, ground-truth visible masks)

| | SAIL-VOS mIoU | SAIL-VOS mIoU<sub>occ</sub> | TAO-Amodal val AP<sub>50</sub> | TAO-Amodal test AP<sub>50</sub> | s / object |
|---|---|---|---|---|---|
| Diffusion-VAS | 77.07 | 55.12 | 89.25 | 88.99 | 11.77 |
| **OccJEPA** | **78.60** | 52.09 | **93.36** | **92.99** | **0.35** |

On the frames that both methods are scored on (occluded frames of SAIL-VOS, a miss counting as zero), OccJEPA reaches
43.68 occluded IoU against 39.98. The TAO-Amodal test split (48,675 frames, built with
`occjepa.build_tao_split`) was never used for any design choice.

## Installation

```bash
git clone <this repository> occjepa && cd occjepa
git clone https://github.com/facebookresearch/vjepa2 third_party/vjepa2 && git -C third_party/vjepa2 checkout 204698b
pip install -r requirements.txt && pip install -e .
python scripts/download_weights.py          # V-JEPA 2.1 ViT-G (Meta, 30 GB) + OccJEPA weights (OCCJEPA_CKPT_URL)
```
Tested with Python 3.12, PyTorch 2.x (CUDA), one GPU with 24 GB (training: 17 GB peak) or 12 GB (inference).
Paths are set in `occjepa/paths.py` (`OCCJEPA_DATA`, `OCCJEPA_CACHE`, `OCCJEPA_RUNS`, `OCCJEPA_WEIGHTS`, `VJEPA2_REPO`).

## Amodal masks for your own video

```bash
python -m occjepa.demo --frames my_video/frames --visible my_video/visible_masks \
    --ckpt weights/occjepa_vitG.pt --out out/ --video
```
One object per call; `visible_masks/` holds the object's visible mask in every frame (same file names as the frames;
e.g. from SAM 2). Writes the amodal mask of every frame, the amodal boxes (which may extend beyond the image) and an
overlay video.

## Evaluation

```bash
bash scripts/prepare_data.sh dvas && bash scripts/prepare_data.sh tao     # + sailvos / movi / kubric
python -m occjepa.eval_tao --annot data/tao_amodal/diffusion_vas_tao_amodal_val.json \
    --track_ids data/tao_amodal/tao_amodal_track_ids_abs2rel_val.json --frames data/tao_amodal/frames \
    --ckpt weights/occjepa_vitG.pt --out results/tao_val
python -m occjepa.eval_tao --annot data/tao_amodal/test_seqs.json --track_ids data/tao_amodal/test_abs2rel.json \
    --frames data/tao_amodal/frames --ckpt weights/occjepa_vitG.pt --out results/tao_test
python -m occjepa.eval_sailvos --ckpt weights/occjepa_vitG.pt --out results/sailvos
python -m occjepa.eval_movi --data movid --mode uncropped --ckpt weights/occjepa_vitG.pt --out results/movid.npz
python -m occjepa.score_movi score --data movid --pred results/movid.npz --which pred2 --out results/movid.json
```
`eval_tao` reports one pass and re-crop (the paper's numbers); `eval_sailvos` exports the Diffusion-VAS prediction
format (scoreable with their official script) and reports mIoU, mIoU<sub>occ</sub> and the fixed-frame occluded IoU.
Kubric: `--data kubric_static|kubric_dynamic`, scored with the A counting.

## Training

```bash
python -m occjepa.train --tag occjepa_vitG          # 6000 steps, batch 8, SAIL-VOS + MOVi-D (50/50)
```
LoRA (rank 16) on the last 12 encoder blocks, the pre-trained predictor fine-tuned, a head with a high-resolution
refiner; about 5 s per step on one GPU (8 h for 6000 steps). Variants of the paper's ablations: `--arm B2_lora_scr` (randomly initialised
predictor), `--arm C_lora` (no predictor), `--world_model vitg`.

## Code map

| file | content |
|---|---|
| `occjepa/model.py` | OccJEPA: occlusion as masking, latent completion with the predictor, mask head |
| `occjepa/layers.py` | mask head with refiner, LoRA, drop sets, predictor fill, token coverage |
| `occjepa/vjepa.py` | V-JEPA 2.1 loading (official code) and predictor forward on any token grid |
| `occjepa/infer.py`, `tao.py` | two-pass inference (re-crop) on the padded canvas, Diffusion-VAS box metric |
| `occjepa/data/` | SAIL-VOS crop caches, MOVi-D, the 50/50 training mixture, SAIL-VOS re-crop |
| `occjepa/train.py` | training |
| `occjepa/eval_*.py`, `score_movi.py` | evaluation protocols |
| `occjepa/sailvos_cache.py`, `convert_*.py`, `build_tao_split.py`, `fetch_tao_frames.py` | data preparation |

## License

MIT (see `LICENSE`). V-JEPA 2 code and weights: Meta, under their licenses; datasets under their own terms.
