"""Kimodo-G1, the motion generator, in one place: LLM2Vec text encoding, the cached-embedding text encoder, model
loading and text-only sampling. Every Kimodo step uses it: prior motions and motion candidates (`samples`, below),
lead-ins (datagen_pipelines/shared/leadin.py) and the HALO-baseline clip and approach (datagen_pipelines/halo_infinigen/). Runs in the kimodo env, from
kimodo/ with PYTHONPATH=<repo> (cwd first, so `kimodo` is the package, not the repo folder).

`samples` reads the `kimodo:` block of its config:
    device    cuda: one process (encode, then generate).
              cpu (login node): LLM2Vec alone holds ~15 GB and with Kimodo on top the login node kills the process, so
              the encoding runs in its own subprocess (writes text_emb.npz and exits), then Kimodo loads with the
              cached embeddings (~3 GB, ~30 s per 5 s motion at 100 steps). Keep <= 6 motions per call there.
    n, subdir, prefix   <prior>/<subdir>/<prefix>00..<n-1>.{csv,npz} (csv: 30 fps, root xyz, wxyz, 29 joints)
    duration, steps, seed   seeds come from the prior's name (+ seed), so array tasks never share a seed stream
    emb_dir   scratch dir for the prompt embeddings (one subdir per array task)

    python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/motions
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import zlib

import numpy as np

from humanoid_datagen.utils.paths import REPO_ROOT as REPO, repo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage
MODEL = "Kimodo-G1-RP-v1"


def rp(p: str) -> str:
    return repo_path(p)


# ------------------------------------------------------------------------------------------------ text
def key(t: str) -> str:
    return t.strip().lower().rstrip(".")  # Kimodo capitalises / punctuates prompts before encoding


def encode(prompts: list[str], out_npz: str, device: str = "cpu", threads: int = 6) -> None:
    """LLM2Vec embeddings of `prompts` -> out_npz (prompts, embs). In-process."""
    import torch
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS
    torch.set_num_threads(threads)
    kw = dict(TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"])
    kw["device"] = device
    kw.pop("_target_", None)
    enc = LLM2VecEncoder(**kw)
    embs = [enc(p)[0].float().cpu().numpy().reshape(-1) for p in prompts]
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    np.savez(out_npz, prompts=np.array(prompts), embs=np.stack(embs))
    print(f"encoded {len(prompts)} prompts -> {out_npz}", flush=True)


def encode_for(prompts: list[str], out_npz: str, device: str) -> None:
    """encode() in-process on cuda; in its own subprocess on cpu, so LLM2Vec's memory is gone before Kimodo loads."""
    if device != "cpu":
        encode(prompts, out_npz, device)
        return
    os.makedirs(os.path.dirname(os.path.abspath(out_npz)), exist_ok=True)
    pj = out_npz[:-4] + "_prompts.json"
    json.dump(prompts, open(pj, "w"))
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (REPO, os.environ.get("PYTHONPATH")) if p))
    subprocess.run([sys.executable, "-m", "humanoid_datagen.datagen_pipelines.shared.kimodo", "encode", pj, out_npz],
                   check=True, env=env)


class CachedText:
    """Kimodo text encoder from cached embeddings. Empty prompts (CFG) and the load healthcheck get zeros, which is
    what Kimodo itself sets for empty prompts; any other unknown prompt is an error."""

    def __init__(self, path):
        z = np.load(path)
        self.d = {key(str(p)): e for p, e in zip(z["prompts"], z["embs"])}
        self.dim = z["embs"].shape[1]

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def get_device(self):
        import torch
        return torch.device("cpu")

    def __call__(self, text):
        import torch
        single = isinstance(text, str)
        texts = [text] if single else list(text)
        rows = []
        for t in texts:
            k = key(t)
            if k in self.d:
                rows.append(self.d[k])
            elif k in ("", "healthcheck"):
                rows.append(np.zeros(self.dim, np.float32))
            else:
                raise KeyError(f"prompt not encoded: {t!r}")
        e = torch.tensor(np.stack(rows))[:, None]
        return (e[0], 1) if single else (e, [1] * len(texts))


def load(text_emb: str, device: str = "cpu", threads: int = 6):
    """(model, qpos converter) with the cached text embeddings."""
    import torch
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    torch.set_num_threads(threads)
    model = load_model(MODEL, device=device, text_encoder=CachedText(text_emb))
    return model, MujocoQposConverter(model.skeleton)


# ------------------------------------------------------------------------------------------------ keyframed samples
def halo_path() -> None:
    """Put HALO's datagen helpers ($THIRD_PARTY_ROOT/ml-halo/datagen: motion_gen, common) on sys.path."""
    from humanoid_datagen.utils.paths import third_party
    p = third_party("ml-halo", "datagen")
    if p not in sys.path:
        sys.path.insert(0, p)


def approach(C, model, skel, conv, plan, stand, clip0, seed, device):
    """Walk-up into a task motion (HALO-baseline approach, lead-ins). One Kimodo call per sample: full-body keyframes HALO standing pose (frame 0) -> clip frame 0 (last frame),
    root path = the unicycle plan on its first 85% (HALO walk_qpos_from_traj rule), then bend_frames free frames
    for the stand -> bend transition. Uses HALO's segment helpers (canonical frame of the start pose)."""
    from motion_gen.kimodo.constraints import build_constraint_object, build_root2d_constraint, canonicalize
    from motion_gen.kimodo.segments import _canonical_endpoints, _sample_world, world_qpos_to_kf
    T = len(plan)
    N = T + int(C["bend_frames"])
    j0c, r0c, j1c, r1c, canon, origin = _canonical_endpoints(stand, clip0, conv)

    def canon_xz(xy):                         # world pelvis xy -> canonical root (x, z), via the same converter
        q = stand.copy()
        q[:2] = xy
        j, r = world_qpos_to_kf(q, conv)
        jc, _ = canonicalize(j, r, canon)
        return np.array([jc[0, 0] - origin[0], jc[0, 2] - origin[1]], np.float32)

    fidx = np.unique(np.linspace(0, int(0.85 * (T - 1)), 20).astype(int))
    fidx = fidx[fidx > 0]                                         # the full-body keyframe owns frame 0
    root = np.stack([canon_xz(plan[i, :2]) for i in fidx])
    clst = [build_root2d_constraint(fidx, root, None, skel).to(device=device),
            build_constraint_object([(0, j0c, r0c), (N - 1, j1c, r1c)], skel).to(device=device)]
    out = _sample_world(model, conv, C["approach_prompt"], N, clst, seed, int(C["steps"]), tuple(C["cfg"]),
                        int(C["walk_batch"]), origin, canon)
    return np.asarray(out[None] if out.ndim == 2 else out, np.float64), T


# ------------------------------------------------------------------------------------------------ text-only samples
def sample(model, conv, prompt: str, frames: int, steps: int, seed: int, stem: str, device: str) -> None:
    """One text-only motion -> stem.{npz,csv}."""
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.tools import seed_everything
    seed_everything(seed)
    out = model([prompt], [frames], constraint_lst=[], num_denoising_steps=steps, num_samples=1,
                multi_prompt=True, post_processing=False, return_numpy=True)
    single = {k: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v) for k, v in out.items()}
    save_kimodo_npz(stem + ".npz", single)
    conv.save_csv(conv.dict_to_qpos(out, device), stem + ".csv")
    print("saved", stem, flush=True)


def samples(K: dict, prior_dirs: list[str], task: str = "0") -> None:
    """`n` motions per prior from its kimodo.txt (skips priors that already have them)."""
    import torch
    n, sub, pre = int(K["n"]), K["subdir"], K["prefix"]
    dev = K["device"]

    def n_done(pd):
        md = os.path.join(pd, sub)
        return len([f for f in os.listdir(md) if f.endswith(".csv")]) if os.path.isdir(md) else 0

    todo = [pd for pd in prior_dirs if n_done(pd) < n]
    if not todo:
        print("nothing to do", flush=True)
        return
    emb = os.path.join(rp(K["emb_dir"]), f"task{task}", "text_emb.npz")
    encode_for([open(os.path.join(pd, "kimodo.txt")).read().strip() for pd in todo], emb, dev)  # once per call
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model, conv = load(emb, dev, threads=int(K.get("threads", 8)))
    T = int(round(float(K["duration"]) * model.fps))
    w = len(str(n - 1))
    for pd in todo:
        md = os.path.join(pd, sub)
        os.makedirs(md, exist_ok=True)
        prompt = open(os.path.join(pd, "kimodo.txt")).read().strip()
        s0 = int(K["seed"]) + 1000 * (zlib.crc32(os.path.basename(os.path.normpath(pd)).encode()) % 100000)
        for k in range(n):
            stem = os.path.join(md, f"{pre}{k:0{w}d}")
            if not os.path.exists(stem + ".csv"):
                sample(model, conv, prompt, T, int(K["steps"]), s0 + k, stem, dev)


class MotionCandidates(Stage):
    """Datagen stage `motions`: `kimodo.n` samples per prior of this array task (shared/priors.py)."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        from humanoid_datagen.datagen_pipelines.shared.priors import prior_dirs
        samples(C["kimodo"], prior_dirs(C), os.environ.get("SLURM_ARRAY_TASK_ID", "0"))


if __name__ == "__main__":       # internal: encode_for's subprocess on cpu
    a = sys.argv[1:]
    if len(a) != 3 or a[0] != "encode":
        raise SystemExit("usage (internal): python -m humanoid_datagen.datagen_pipelines.shared.kimodo encode <prompts.json> <out.npz>")
    encode(json.load(open(a[1])), a[2], "cpu")
