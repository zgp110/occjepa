"""Where data, caches and runs live (override with environment variables)."""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.environ.get("OCCJEPA_DATA", os.path.join(ROOT, "data"))        # raw / converted datasets
CACHE = os.environ.get("OCCJEPA_CACHE", os.path.join(ROOT, "cache"))     # SAIL-VOS crop caches
RUNS = os.environ.get("OCCJEPA_RUNS", os.path.join(ROOT, "runs"))        # training runs (checkpoints, logs)

SAILVOS_RAW = os.path.join(DATA, "sailvos")            # extracted SAIL-VOS-2D (see scripts/prepare_sailvos.sh)
SAILVOS_CACHE = os.path.join(CACHE, "sailvos")         # cache_<split>/ crop caches (build_cache)
DVAS_ANNOT = os.path.join(DATA, "diffusion_vas")       # Diffusion-VAS sequence jsons (HF kaihuac/diffusion_vas_datasets)
MOVI = {"movid": os.path.join(DATA, "movi", "movid"), "movib": os.path.join(DATA, "movi", "movib")}
KUBRIC = {"kubric_static": os.path.join(DATA, "kubric_taco", "static", "test"),
          "kubric_dynamic": os.path.join(DATA, "kubric_taco", "dynamic", "test")}
TAO = os.path.join(DATA, "tao_amodal")                 # frames/ and the sequence jsons
