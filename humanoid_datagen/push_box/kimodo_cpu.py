"""Kimodo-G1 push motions. GPU: `samples` mode (one process per batch of priors). CPU-only node (login node): two
light processes, as below.

The LLM2Vec text encoder alone holds ~15 GB; with Kimodo on top the della login node kills the
process. So: `encode` the prompts once (writes text_emb.npz, then exits), then `generate` with the
cached embeddings (~3 GB, ~30 s per 5 s motion at 100 steps).

Run from kimodo/ in the kimodo env:
    python ../humanoid_datagen/push_box/kimodo_cpu.py encode   <out_dir> A.txt B.txt ...
    python ../humanoid_datagen/push_box/kimodo_cpu.py generate <out_dir> --job A.txt A0 --job A.txt A1 ...
Output: <out_dir>/<stem>.{npz,csv} as in kimodo.scripts.generate_batch (csv: 30 fps, root xyz, wxyz, 29 joints).
"""
import argparse
import os

import numpy as np
import torch


def encode(out_dir, prompt_files, device="cpu"):
    from kimodo.model.llm2vec.llm2vec_wrapper import LLM2VecEncoder
    from kimodo.model.load_model import TEXT_ENCODER_PRESETS

    kw = dict(TEXT_ENCODER_PRESETS["llm2vec"]["kwargs"])
    kw["device"] = device
    kw.pop("_target_", None)
    enc = LLM2VecEncoder(**kw)
    prompts = [open(p).read().strip() for p in prompt_files]
    embs = [enc(p)[0].float().cpu().numpy().reshape(-1) for p in prompts]
    np.savez(os.path.join(out_dir, "text_emb.npz"), prompts=np.array(prompts), embs=np.stack(embs))
    print("encoded", len(prompts))


def _key(t):
    return t.strip().lower().rstrip(".")  # Kimodo capitalises / punctuates prompts before encoding


class Cached:
    def __init__(self, d):
        self.d = {_key(k): v for k, v in d.items()}

    def to(self, *a, **k):
        return self

    def eval(self):
        return self

    def get_device(self):
        return torch.device("cpu")

    def __call__(self, text):
        single = isinstance(text, str)
        texts = [text] if single else text
        e = torch.tensor(np.stack([self.d[_key(t)] for t in texts]))[:, None]
        return (e[0], 1) if single else (e, [1] * len(texts))


def generate(out_dir, jobs, duration, steps, seed, device="cpu", emb_dir=None, model=None):
    from kimodo import load_model
    from kimodo.exports.motion_io import save_kimodo_npz
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.tools import seed_everything

    z = np.load(os.path.join(emb_dir or out_dir, "text_emb.npz"))
    d = {str(p): e for p, e in zip(z["prompts"], z["embs"])}
    if model is None:
        model = load_model("Kimodo-G1-RP-v1", device=device, text_encoder=Cached(d))
    conv = MujocoQposConverter(model.skeleton)
    T = int(round(duration * model.fps))
    for k, (pf, stem) in enumerate(jobs):
        prompt = open(pf).read().strip()
        if seed is not None:
            seed_everything(seed + k)
        out = model([prompt], [T], constraint_lst=[], num_denoising_steps=steps, num_samples=1,
                    multi_prompt=True, post_processing=False, return_numpy=True)
        single = {kk: (v[0] if hasattr(v, "shape") and len(v.shape) > 0 and v.shape[0] == 1 else v) for kk, v in out.items()}
        stem = os.path.join(out_dir, stem)
        save_kimodo_npz(stem + ".npz", single)
        conv.save_csv(conv.dict_to_qpos(out, device), stem + ".csv")
        print("saved", stem, flush=True)
    return model


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["encode", "generate", "samples"])
    ap.add_argument("out_dir")
    ap.add_argument("prompts", nargs="*")
    ap.add_argument("--job", nargs=2, action="append", default=[])
    ap.add_argument("--duration", type=float, default=5.0)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=4, help="samples mode: motions per prompt file")
    ap.add_argument("--subdir", default="motions", help="samples mode: <prior>/<subdir>/<prefix>NN.csv")
    ap.add_argument("--prefix", default="m")
    a = ap.parse_args()
    torch.set_num_threads(8)
    if a.mode == "encode":
        encode(a.out_dir, a.prompts, a.device)
    elif a.mode == "generate":
        generate(a.out_dir, a.job, a.duration, a.steps, a.seed, a.device)
    else:  # samples: out_dir = embedding scratch dir; prompts = prior dirs -> <prior>/<subdir>/<prefix>0..<n-1>.csv
        import zlib

        def n_done(pd):
            md = os.path.join(pd, a.subdir)
            return len([f for f in os.listdir(md) if f.endswith(".csv")]) if os.path.isdir(md) else 0

        todo = [pd for pd in a.prompts if n_done(pd) < a.n]
        if todo:
            os.makedirs(a.out_dir, exist_ok=True)
            encode(a.out_dir, [os.path.join(pd, "kimodo.txt") for pd in todo], a.device)  # once (LLM2Vec ~15 GB)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            model = None
            w = len(str(a.n - 1))
            for pd in todo:
                md = os.path.join(pd, a.subdir)
                os.makedirs(md, exist_ok=True)
                pf = os.path.join(pd, "kimodo.txt")
                # seeds from the prior's name, so priors in different array tasks never share a seed stream
                s0 = a.seed + 1000 * (zlib.crc32(os.path.basename(os.path.normpath(pd)).encode()) % 100000)
                model = generate(md, [(pf, f"{a.prefix}{k:0{w}d}") for k in range(a.n)], a.duration, a.steps, s0,
                                 a.device, emb_dir=a.out_dir, model=model)
