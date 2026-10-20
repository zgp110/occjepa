#!/usr/bin/env bash
# Data for training and evaluation (paths: occjepa/paths.py; override with OCCJEPA_DATA / OCCJEPA_CACHE).
# Run the parts you need, from the repository root, e.g.  bash scripts/prepare_data.sh tao
set -euo pipefail
DATA=${OCCJEPA_DATA:-$PWD/data}; CACHE=${OCCJEPA_CACHE:-$PWD/cache}
PY=${PYTHON:-python}
case ${1:-help} in
dvas)    # Diffusion-VAS sequence jsons (SAIL-VOS train / val, TAO-Amodal val) from HF kaihuac/diffusion_vas_datasets
  git clone https://huggingface.co/datasets/kaihuac/diffusion_vas_datasets $DATA/diffusion_vas;;
sailvos) # needs the official SAIL-VOS RGB archives (request access at sailvos.web.illinois.edu), extracted with the
         # official recipe into $DATA/sailvos/{train,val}_raw:  cat sailvos_<split>.tar.* | tar xf - && unzip sailvos_<split>.zip
  D=$DATA/diffusion_vas
  $PY -m occjepa.sailvos_cache --split svtrain_grid --seq_json $D/diffusion_vas_sailvos_train.json \
      --rgb_root $DATA/sailvos/train_raw --root $CACHE/sailvos --resample grid --no_pad \
      --exclude occjepa/resources/sailvos_train_excluded_cross_video.json
  $PY -m occjepa.sailvos_cache --split svval_grid --seq_json $D/diffusion_vas_sailvos_val.json \
      --rgb_root $DATA/sailvos/val_raw --root $CACHE/sailvos --resample grid --no_pad;;
movi)    # EoRaS's MOVi-D (21.5 GB) and MOVi-B (27.1 GB) releases
  mkdir -p $DATA/movi/archives
  for n in MOVi-D MOVi-B; do
    curl -L --fail -C - --retry 5 -o $DATA/movi/archives/$n.tar.gz https://data.dgl.ai/dataset/EoRas/$n.tar.gz
  done
  $PY -m occjepa.convert_movi --tar $DATA/movi/archives/MOVi-D.tar.gz --out $DATA/movi/movid
  $PY -m occjepa.convert_movi --tar $DATA/movi/archives/MOVi-B.tar.gz --out $DATA/movi/movib;;
kubric)  # TACO's Kubric-Static / Kubric-Dynamic benchmarks (HF JasonAplp/TACO)
  mkdir -p $DATA/kubric_taco/archives
  for f in kubric_static kubric_dynamic; do
    curl -L --fail -C - --retry 5 -o $DATA/kubric_taco/archives/$f.zip \
      https://huggingface.co/datasets/JasonAplp/TACO/resolve/main/benchmarks/$f.zip
    $PY -m occjepa.convert_kubric --zip $DATA/kubric_taco/archives/$f.zip --root $f --out $DATA/kubric_taco/${f#kubric_}
  done;;
tao)     # TAO-Amodal (gated on HF chengyenhsieh/TAO-Amodal: accept the terms; token in ~/.cache/huggingface/token).
         # Annotations into $DATA/tao_amodal/annot (BURST_annotations/, amodal_annotations/), then our test split and the
         # frames of the val (Diffusion-VAS) and test sequences.
  T=$DATA/tao_amodal
  $PY -m occjepa.build_tao_split --split test --annot_root $T/annot --out_dir $T
  cp $DATA/diffusion_vas/diffusion_vas_tao_amodal_val.json $DATA/diffusion_vas/tao_amodal_track_ids_abs2rel_val.json $T/
  $PY -m occjepa.fetch_tao_frames --annot $T/diffusion_vas_tao_amodal_val.json --out $T/frames
  $PY -m occjepa.fetch_tao_frames --annot $T/test_seqs.json --out $T/frames;;
*) echo "usage: bash scripts/prepare_data.sh dvas|sailvos|movi|kubric|tao"; exit 1;;
esac
