"""Manual server entry: build frozen FP32 train/test task caches, then train."""

import argparse
import gc
from pathlib import Path
import sys

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from cscwd.code.data_prep.task_cache import TaskFeatureCache, build_cache, sha256
from cscwd.code.scripts.train_convlstm import add_training_args, config_from_args, run_training, seed_everything


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train-tracks", required=True)
    parser.add_argument("--test-tracks", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--reuse-cache", action="store_true",
                        help="Read existing completed matching caches; never overwrite them")
    add_training_args(parser)
    args = parser.parse_args()
    # GazeSpot's existing backbone factory resolves local weights from cwd.
    if Path.cwd().resolve() != REPO_ROOT:
        raise ValueError(f"Launch from the repository root: {REPO_ROOT}")
    if Path(args.output_dir).exists() or Path(args.report_dir).exists():
        raise FileExistsError("Choose fresh experiment output and report directories")
    seed_everything(args.seed)
    cache_root = Path(args.cache_root)
    for split, tracks in (("train", args.train_tracks), ("test", args.test_tracks)):
        cache_dir = cache_root / split
        if args.reuse_cache and (cache_dir / "index.json").is_file():
            cache = TaskFeatureCache(cache_dir)
            source = cache.index["source"]
            if (cache.index["split"] != split or source["tracks_sha256"] != sha256(tracks)
                    or source["checkpoint_sha256"] != sha256(args.checkpoint)
                    or Path(source["data_root"]).resolve() != Path(args.data_root).resolve()):
                raise ValueError(f"Existing {split} cache does not match requested inputs")
            print(f"Reusing completed {split} cache: {cache_dir}", flush=True)
        else:
            build_cache(args.data_root, tracks, args.checkpoint, cache_dir, split,
                        args.device, args.cache_batch_size, args.num_workers)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    run_training(cache_root / "train", cache_root / "test", args.output_dir, args.report_dir,
                 args.provenance, config_from_args(args))


if __name__ == "__main__":
    main()
