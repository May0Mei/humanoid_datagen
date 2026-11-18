"""Tiny JSON ledger for generation cost: GPU-hours and human-hours per source/stage."""
from __future__ import annotations

import fcntl
import json
import os
import time


class CostLedger:
    """Ledger file: {"entries": [{source, stage, gpu_hours, human_hours, n_scenes, note, time}]}.
    Stages are free strings, conventionally: scene_gen, demo_gen, training, eval."""

    def __init__(self, path: str):
        self.path = path

    def _load(self) -> dict:
        if not os.path.exists(self.path):
            return {"entries": []}
        with open(self.path) as f:
            return json.load(f)

    def log(self, source: str, stage: str, gpu_hours: float = 0.0, human_hours: float = 0.0,
            n_scenes: int | None = None, note: str = "") -> None:
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        with open(self.path + ".lock", "w") as lk:
            fcntl.flock(lk, fcntl.LOCK_EX)
            data = self._load()
            data["entries"].append({"source": source, "stage": stage, "gpu_hours": float(gpu_hours),
                                    "human_hours": float(human_hours), "n_scenes": n_scenes,
                                    "note": note, "time": time.time()})
            tmp = self.path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=1)
            os.replace(tmp, self.path)

    def totals(self) -> dict:
        """{source: {stage: {gpu_hours, human_hours, n_scenes}}}; n_scenes = max logged for that stage."""
        out: dict = {}
        for e in self._load()["entries"]:
            d = out.setdefault(e["source"], {}).setdefault(e["stage"], {"gpu_hours": 0.0, "human_hours": 0.0, "n_scenes": None})
            d["gpu_hours"] += e["gpu_hours"]
            d["human_hours"] += e["human_hours"]
            if e.get("n_scenes") is not None:
                d["n_scenes"] = max(d["n_scenes"] or 0, e["n_scenes"])
        return out

    def per_scene(self, stages=("scene_gen",)) -> dict:
        """{source: {gpu_hours, human_hours}} per scene, summed over `stages`."""
        out = {}
        for src, st in self.totals().items():
            g = sum(st[s]["gpu_hours"] for s in stages if s in st)
            h = sum(st[s]["human_hours"] for s in stages if s in st)
            n = max([st[s]["n_scenes"] or 0 for s in stages if s in st] or [0])
            out[src] = {"gpu_hours": g / n if n else None, "human_hours": h / n if n else None, "n_scenes": n}
        return out


class timed_stage:
    """Context manager: `with timed_stage(ledger, "ours", "scene_gen", n_gpus=1, n_scenes=10): ...`"""

    def __init__(self, ledger: CostLedger, source: str, stage: str, n_gpus: float = 1.0,
                 n_scenes: int | None = None, human_hours: float = 0.0, note: str = ""):
        self.args = (ledger, source, stage, n_gpus, n_scenes, human_hours, note)

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        ledger, source, stage, n_gpus, n_scenes, human_hours, note = self.args
        ledger.log(source, stage, gpu_hours=n_gpus * (time.time() - self.t0) / 3600,
                   human_hours=human_hours, n_scenes=n_scenes, note=note)
