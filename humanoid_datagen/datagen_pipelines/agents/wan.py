"""Wan2.2 T2V-A14B text-to-video: prompt file(s) -> mp4, loading the two MoE experts once for all jobs."""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

# The Wan2.2 repo lives in $THIRD_PARTY_ROOT/Wan2.2 (default <repo>/utilities/repos, see README).
sys.path.insert(0, os.path.join(os.environ.get("THIRD_PARTY_ROOT", str(Path(__file__).resolve().parents[3] / "utilities" / "repos")),
                                "Wan2.2"))

import torch
from huggingface_hub.constants import HF_HUB_CACHE

import wan
from wan.configs import SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
from wan.utils.utils import save_video
from humanoid_datagen.datagen_pipelines.shared.base import Stage

MODEL_REPO = "Wan-AI/Wan2.2-T2V-A14B"
TASK = "t2v-A14B"


def resolve_local(model):
    """Map a repo id to its cached snapshot dir (compute nodes are offline); pass local dirs through."""
    if Path(model).is_dir():
        return model
    repo_cache = Path(HF_HUB_CACHE) / f"models--{model.replace('/', '--')}"
    commit = (repo_cache / "refs" / "main").read_text().strip()
    return str(repo_cache / "snapshots" / commit)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--prompt-file")
    p.add_argument("--out")
    # Batch mode: repeat --job to run several prompts with one model load.
    p.add_argument("--job", nargs=2, action="append", default=[], metavar=("PROMPT_FILE", "OUT"))
    p.add_argument("--model", default=MODEL_REPO, help="repo id (resolved from the HF cache) or local checkpoint dir")
    p.add_argument("--size", default="1280*720", choices=SUPPORTED_SIZES[TASK])
    # 81 frames at 16 fps = 5 s; frame count must be 4n+1.
    p.add_argument("--frame-num", type=int, default=81)
    p.add_argument("--seed", type=int, default=42)
    # Several seeds: every job runs once per seed and writes <out stem>_seed<N><ext> instead of <out>.
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="run each job once per seed (overrides --seed)")
    p.add_argument("--skip-existing", action="store_true", help="skip outputs that already exist (resume a timed-out run)")
    p.add_argument("--steps", type=int, default=None, help="sampling steps; default = config (40)")
    p.add_argument("--neg-prompt", default="", help="negative prompt; default = Wan's built-in one")
    # The checkpoints are fp32 (54GB per expert). Cast to bf16 and keep both experts + T5 on the GPU (~65GB),
    # which fits a 140GB H200. --offload restores upstream's single-GPU behavior (swap experts via CPU).
    p.add_argument("--offload", action="store_true", help="offload experts/T5 to CPU between uses (for <=80GB GPUs)")
    args = p.parse_args(argv)

    jobs = list(args.job)
    if args.prompt_file or args.out:
        if not (args.prompt_file and args.out):
            p.error("--prompt-file and --out go together")
        jobs.insert(0, (args.prompt_file, args.out))
    if not jobs:
        p.error("give --prompt-file/--out or at least one --job")
    if (args.frame_num - 1) % 4:
        p.error("--frame-num must be 4n+1")

    # Expand to one run per (job, seed).
    if args.seeds:
        runs = [(pf, str(Path(o).with_name(f"{Path(o).stem}_seed{s}{Path(o).suffix}")), s) for pf, o in jobs for s in args.seeds]
    else:
        runs = [(pf, o, args.seed) for pf, o in jobs]
    if args.skip_existing:
        skipped = [r for r in runs if Path(r[1]).exists()]
        runs = [r for r in runs if not Path(r[1]).exists()]
        print(f"skipping {len(skipped)} existing outputs", flush=True)
    if not runs:
        return

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s", stream=sys.stdout)
    cfg = WAN_CONFIGS[TASK]
    pipe = load_pipeline(args, cfg)
    failed = []
    for i, (prompt_file, out, seed) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {prompt_file} (seed {seed}) -> {out}", flush=True)
        try:
            generate(pipe, cfg, args, prompt_file, out, seed)
        except Exception as e:  # keep going so one bad job doesn't waste the loaded model
            print(f"FAILED {out}: {e!r}", flush=True)
            failed.append(out)
    if failed:
        raise SystemExit(f"{len(failed)}/{len(runs)} jobs failed: {failed}")


def load_pipeline(args, cfg):
    t0 = time.time()
    pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=resolve_local(args.model),
        device_id=0,
        rank=0,
        init_on_cpu=args.offload,
        convert_model_dtype=True,
    )
    print(f"loaded {TASK} in {time.time() - t0:.0f}s", flush=True)
    return pipe


def generate(pipe, cfg, args, prompt_file, out, seed):
    prompt = Path(prompt_file).read_text().strip()

    t0 = time.time()
    video = pipe.generate(
        prompt,
        size=SIZE_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=cfg.sample_shift,
        sample_solver="unipc",
        sampling_steps=args.steps or cfg.sample_steps,
        guide_scale=cfg.sample_guide_scale,
        n_prompt=args.neg_prompt,
        seed=seed,
        offload_model=args.offload,
    )
    print(f"generation took {time.time() - t0:.0f}s")

    Path(out).parent.mkdir(parents=True, exist_ok=True)
    save_video(tensor=video[None], save_file=out, fps=cfg.sample_fps, nrow=1, normalize=True, value_range=(-1, 1))
    print(f"wrote {out}")


class Videos(Stage):
    """Datagen stage `videos` (Wan2.2 env): one video per prior of this array task (shared/priors.py), from its
    wan.txt, named after its camera (wan_<third|ego>_seed<s>.mp4); existing videos are skipped."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        import json
        from humanoid_datagen.datagen_pipelines.shared.priors import prior_dirs
        argv = []
        for pd in prior_dirs(C):
            cam = json.load(open(os.path.join(pd, "meta.json")))["camera"]
            argv += ["--job", os.path.join(pd, "wan.txt"), os.path.join(pd, f"wan_{cam}.mp4")]
        if not argv:
            print("no priors for this task")
            return
        argv += ["--seeds", *map(str, C["seeds"]), "--skip-existing", "--frame-num", str(C["frame_num"]), "--size", C["size"]]
        main(argv + (["--offload"] if C.get("offload") else []))


if __name__ == "__main__":
    main()
