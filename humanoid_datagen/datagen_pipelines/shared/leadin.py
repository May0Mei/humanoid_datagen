"""Standing lead-in for every selected push motion (Training datagen v2, step 3; docs/push_box.md).

The eval starts every episode from SONIC's default standing pose, so each training demo starts there too: one Kimodo
call per motion, the way the HALO arm makes its approach (shared/kimodo.approach, HALO's segment helpers):
  * full-body keyframes: frame 0 = SONIC's standing pose (evaluators/grscenes.stand_pose joints, feet on the floor), the last
    frame = the motion's frame 0 (the start of the bend / squat);
  * root path: a straight line, `travel` m of pelvis travel along the motion's push direction, at HALO's walk speed;
    `bend_frames` free frames after it for the stand -> bend;
  * prompt "a person walks up to a <object> and bends over to push it." (the prior's object);
  * best of `walk_batch` samples by HALO's push_box motion gate on the whole motion (fewest blocking checks, then
    score), joined with HALO's seam_blend.
The lead-in is scene-free: the datagen moves lead-in ++ motion rigidly to each spot and checks clearance there.

Per prior <root>/<pid>/leadin/: spec.json (prep), m<k>.csv = lead-in ++ motion (30 fps, Kimodo / robot qpos layout,
first frame = the standing pose), m<k>.json (motion_start frame, gate verdict, pick).

One run (kimodo env, GPU; the env also has MuJoCo for prep): prep (push direction and standing start per motion),
encode (prompt embeddings), gen (Kimodo samples, gate, seam). Array task i = priors chunk i.
  python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/leadin
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from humanoid_datagen.utils.paths import REPO_ROOT as REPO, repo_path
from humanoid_datagen.datagen_pipelines.shared.base import Stage
FPS = 30.0


def rp(p: str) -> str:
    return repo_path(p)


def priors(C, task: int | None = None) -> list[str]:
    """Priors with motions (optionally `priors`), or array task `task`'s chunk of `per_task` of them."""
    pds = sorted(d for d in glob.glob(os.path.join(rp(C.get("root") or C["priors_root"]), "p[0-9][0-9][0-9]"))
                 if glob.glob(os.path.join(d, "motions", "m*.csv")))
    if C.get("priors"):
        pds = [d for d in pds if os.path.basename(d) in set(C["priors"])]
    if task is not None:
        n = int(C["per_task"])
        pds = pds[task * n:(task + 1) * n]
    return pds


def prompt_of(pd, tk) -> str:
    return tk.approach_prompt(json.load(open(os.path.join(pd, "meta.json")))["object"])


# ------------------------------------------------------------------------------------------------ prep (mujoco env)
def stand_qpos(model, xy, yaw) -> np.ndarray:
    """SONIC's default standing pose (evaluators/grscenes.stand_pose) at xy facing yaw, lowest robot geom 5 mm above z = 0,
    in the motion CSV layout (= the robot model's 36-d qpos)."""
    import mujoco
    from humanoid_datagen.utils.sonic import DEFAULT, JOINT_NAMES, robot_free_joint
    d = mujoco.MjData(model)
    qa = model.jnt_qposadr[robot_free_joint(model).id]
    assert qa == 0 and model.nq == 36
    d.qpos[:3] = [xy[0], xy[1], 1.0]
    d.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    for k, n in enumerate(JOINT_NAMES):
        d.qpos[model.jnt_qposadr[model.joint(n).id]] = DEFAULT[k]
    mujoco.mj_kinematics(model, d)
    zs = []
    for g in range(model.ngeom):
        if model.geom_bodyid[g] > 0 and (model.geom_contype[g] or model.geom_conaffinity[g]):  # robot, not floor
            c, R, a = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3), model.geom_aabb[g]
            zs.append(c[2] + R[2] @ a[:3] - np.abs(R[2]) @ a[3:])
    q = d.qpos.copy()
    q[2] -= min(zs) - 0.005
    return q


def prep(C, tk, task=None):
    """tk: the task module (its staging function finds each motion's push direction)."""
    from humanoid_datagen.datagen_pipelines.shared.compose import robot_only_model
    from humanoid_datagen.utils.sonic import Reference
    rm = robot_only_model()
    travel = float(C["travel"])
    for pd in priors(C, task):
        out = os.path.join(pd, "leadin")
        os.makedirs(out, exist_ok=True)
        spec = dict(prompt=prompt_of(pd, tk), travel=travel, motions={})
        for csv in sorted(glob.glob(os.path.join(pd, "motions", "m*.csv"))):
            Q = np.loadtxt(csv, delimiter=",")
            u = np.asarray(tk.motion_frame(rm, Reference.from_qpos(Q, FPS), [0.1, 0.1, 0.1])["dir"][:2], float)
            u /= np.linalg.norm(u)
            yaw = float(np.arctan2(u[1], u[0]))
            xy = Q[0, :2] - travel * u
            spec["motions"][os.path.basename(csv)[:-4]] = dict(
                stand=stand_qpos(rm, xy, yaw).tolist(), direction=u.tolist(), yaw_rad=yaw)
        json.dump(spec, open(os.path.join(out, "spec.json"), "w"), indent=1)
        print(f"[prep] {os.path.basename(pd)}: {len(spec['motions'])} motions, '{spec['prompt']}'", flush=True)


# ------------------------------------------------------------------------------------------------ kimodo env
def gen(C, task: int | None, device: str, gate_task: str):
    import torch
    from kimodo import load_model
    from kimodo.exports.mujoco import MujocoQposConverter
    from kimodo.skeleton import G1Skeleton34
    from humanoid_datagen.datagen_pipelines.shared.kimodo import approach, halo_path as _halo_path
    from humanoid_datagen.datagen_pipelines.shared.kimodo import CachedText
    _halo_path()
    from motion_gen.kimodo.motion_gen import seam_blend
    from motion_gen.motion_check.config import CheckConfig
    from motion_gen.motion_check.episode import MotionEpisode
    from motion_gen.motion_check.run import run_episode

    torch.set_num_threads(6)
    pds = priors(C, task)
    model = load_model("Kimodo-G1-RP-v1", device=device, text_encoder=CachedText(rp(C["text_emb"])))
    skel = G1Skeleton34()
    conv = MujocoQposConverter(G1Skeleton34())
    cfg = CheckConfig.for_task(gate_task, tier=C["tier"])

    def gate(q, phases, name):
        rep = run_episode(MotionEpisode(episode_id=name, qpos=q, task_type=gate_task,
                                        meta={"task_type": gate_task, "phases": phases}), cfg, use_mujoco=True)
        return dict(accept=bool(rep.verdict.accept), blocking=list(rep.verdict.blocking),
                    score=float(rep.verdict.score))

    for pd in pds:
        pid = os.path.basename(pd)
        out = os.path.join(pd, "leadin")
        spec = json.load(open(os.path.join(out, "spec.json")))
        Ca = dict(C, approach_prompt=spec["prompt"])
        for mk, ms in spec["motions"].items():
            if os.path.exists(os.path.join(out, mk + ".json")):
                continue
            clip = np.loadtxt(os.path.join(pd, "motions", mk + ".csv"), delimiter=",")
            stand = np.asarray(ms["stand"], np.float64)
            a, b = stand[:2], clip[0, :2]
            T = max(2, int(np.ceil(np.linalg.norm(b - a) / float(C["walk_speed"]) * FPS)) + 1)
            s = np.linspace(0, 1, T)[:, None]
            plan = np.concatenate([a + s * (b - a), np.full((T, 1), ms["yaw_rad"])], axis=1)
            seed = int(C["seed"]) * 100000 + int(pid[1:]) * 100 + int(mk[1:])
            apps, T = approach(Ca, model, skel, conv, plan, stand, clip[0], seed, device)
            cands = []
            for i, ap in enumerate(apps):
                cat = np.concatenate([ap[:-1], clip], axis=0)   # the last lead-in frame is the motion's frame 0
                q = seam_blend(cat.copy(), [len(ap) - 1], k=int(C["seam_k"]))
                ph = {"WALK": [0, T], "BRIDGE": [T, len(ap) - 1], "PUSH": [len(ap) - 1, len(q)]}
                cands.append((q, ph, gate(q, ph, f"{pid}_{mk}_s{i}")))
            wi = min(range(len(cands)), key=lambda i: (len(cands[i][2]["blocking"]), -cands[i][2]["score"]))
            q, ph, v = cands[wi]
            np.savetxt(os.path.join(out, mk + ".csv"), q, delimiter=",", fmt="%.9g")
            meta = dict(motion=f"motions/{mk}.csv", motion_start=int(ph["PUSH"][0]), phases=ph, frames=len(q),
                        fps=FPS, pick=int(wi), batch=len(cands), n_accept=int(sum(c[2]["accept"] for c in cands)),
                        gate=v, seed=seed, prompt=spec["prompt"], travel=spec["travel"])
            json.dump(meta, open(os.path.join(out, mk + ".json"), "w"), indent=1)
            print(f"[gen] {pid} {mk}: lead-in {ph['PUSH'][0]} frames ({ph['PUSH'][0] / FPS:.2f} s), pick {wi}, "
                  f"{meta['n_accept']}/{len(cands)} pass the gate; blocking={v['blocking']}", flush=True)


class LeadIns(Stage):
    """Datagen stage `leadin` (kimodo env): prep, encode and generate for the priors of this array task
    (per_task chunk of SLURM_ARRAY_TASK_ID; all priors without it). Each chunk keeps its own text embeddings."""

    def run(self) -> None:
        C, tk = self.cfg, self.task
        from humanoid_datagen.datagen_pipelines.shared.kimodo import encode_for
        task = int(os.environ["SLURM_ARRAY_TASK_ID"]) if "SLURM_ARRAY_TASK_ID" in os.environ else None
        prep(C, tk, task)
        emb = rp(C["text_emb"])
        if task is not None:
            emb = emb[:-4] + f"_task{task}.npz"
        encode_for(sorted({prompt_of(pd, tk) for pd in priors(C, task)}), emb, C["device"])
        gen(dict(C, text_emb=emb), task, C["device"], tk.halo_gate)
