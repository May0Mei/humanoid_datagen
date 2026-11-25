"""Closed-loop push evaluation in held-out scenes (GRScenes = T-third): planner + GEAR-SONIC + full physics.

    python humanoid_datagen/scripts/eval.py --config_name eval/push/grscenes_halo_<arm>      (GPU node for HALO)

Per test scene (room composed without the robot by the scene composer) a fixed task list is drawn once
(tasks/push/test_objects.py) and written to `tasks.file`, so every policy sees the same episodes: the test object
(the home's own furniture made free-moving, or a held-out library asset placed by the independent sampler;
mass / friction from the test-only ranges, held-out seed), robot start, push direction, gap.

Episode: the G1 stands in SONIC's default pose `gap` m behind the object, facing it. Every `replan_s` the planner
gets the oak1 frame + instruction + proprio and returns a world-frame chunk; SONIC tracks the active chunk
(50 Hz policy, 200 Hz PD). Success (fixed before any run; raw trajectories are kept for rescoring):
    object displaced >= success.min_push m along the push direction at the end, object tilt <= max_tilt_deg
    (measured from the object's pose after the settle, so objects that rest leaning start at 0; the tilt from
    vertical is kept as max_tilt_abs_deg),
    robot never fell (head above run.FALL_HEAD_Z throughout), within timeout_s. Thresholds: the task's pass rule (task.pass_rule).
Ego camera: `ego.calibrated` = the training renders' camera (utils/oak_cam: calibrated OAK-1, HALO's pinhole ->
Kannala-Brandt fisheye, 1280x800); otherwise the robot model's oak1 at ego.width x ego.height.
Planners: halo (vla/halo.HaloPlanner, unmodified) | hold (stand still: the floor of the metric).
"""
from __future__ import annotations

import json
import os
import time

import mujoco
import numpy as np

from humanoid_datagen.vla.halo import MotionChunkCommand
from humanoid_datagen.datagen_pipelines.shared.run import FALL_HEAD_Z, HEAD_BODY
from humanoid_datagen.utils.sonic import DEFAULT, SonicTracker, qmat, robot_free_joint, JOINT_NAMES
from humanoid_datagen.utils.paths import repo_path

DECIM = 4


# ------------------------------------------------------------------------------------------------ tasks
def load_tasks(C: dict) -> list[dict]:
    """The fixed test set (C["tasks"]["file"]); build it first with eval/push/build_grscenes (mode `build`)."""
    path = repo_path(C["tasks"]["file"])
    if not os.path.exists(path):
        raise SystemExit(f"no test set at {C['tasks']['file']}: build it with eval/<task>/build_grscenes")
    return [json.loads(l) for l in open(path) if l.strip()]


def build(C: dict, tk) -> None:
    """Eval mode `build`: draw the task's GRScenes test set (task.build_tasks) and write C["tasks"]["file"]."""
    path = repo_path(C["tasks"]["file"])
    if os.path.exists(path):
        raise SystemExit(f"{C['tasks']['file']} exists: the test set is drawn once (move it away to redraw)")
    out = tk.build_tasks(C)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for t in out:
            f.write(json.dumps(t) + "\n")
    print(f"{len(out)} tasks -> {C['tasks']['file']}")


# ------------------------------------------------------------------------------------------------ episode
class HoldPlanner:
    context_offsets_s = [0.0]
    info = dict(planner="hold")

    def reset(self, task, seed):
        pass

    def act(self, obs):
        if not obs["replan"]:
            return {"chunk": None}
        return {"chunk": MotionChunkCommand(np.asarray(obs["proprio_qpos"])[None], t0=obs["time"], fps=30.0)}


def stand_pose(model, task) -> np.ndarray:
    """36-d qpos: SONIC default joints, facing the push direction, feet on the floor."""
    d = mujoco.MjData(model)
    fj = robot_free_joint(model)
    qa = model.jnt_qposadr[fj.id]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    from humanoid_datagen.datagen_pipelines.shared.compose import episode_frame
    xy, yaw_deg = episode_frame(task)
    y = np.radians(yaw_deg)
    d.qpos[qa:qa + 3] = [*xy, 1.0]
    d.qpos[qa + 3:qa + 7] = [np.cos(y / 2), 0, 0, np.sin(y / 2)]
    d.qpos[qadr] = DEFAULT
    mujoco.mj_kinematics(model, d)
    mujoco.mj_collision(model, d)
    rob = model.geom_contype == 2
    zs = []
    for g in np.where(rob)[0]:
        c = d.geom_xpos[g]
        R = d.geom_xmat[g].reshape(3, 3)
        a = model.geom_aabb[g]
        zs.append(c[2] + R[2] @ a[:3] - np.abs(R[2]) @ a[3:])
    q = np.zeros(36)
    q[:7] = d.qpos[qa:qa + 7]
    q[2] -= min(zs) - 0.005
    q[7:] = DEFAULT
    return q


SETTLE_S = 1.0   # object settle with the robot held still, before the episode (was 0.3 s: objects still rocking)


def place_robot(model, data, task) -> tuple[np.ndarray, int, int, np.ndarray, np.ndarray]:
    """Robot in its standing start pose; returns (q0, free qpos adr, free dof adr, joint qpos adrs, joint dof adrs)."""
    q0 = stand_pose(model, task)
    fj = robot_free_joint(model)
    qa, da = model.jnt_qposadr[fj.id], model.jnt_dofadr[fj.id]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
    data.qpos[qa:qa + 7] = q0[:7]
    data.qpos[qadr] = q0[7:]
    mujoco.mj_forward(model, data)
    return q0, qa, da, qadr, dadr


def settle_held(model, data, q0, qa, da, qadr, dadr, seconds: float, on_step=None) -> None:
    """Step physics `seconds` with the robot pinned at q0 (the object settles alone); velocities zeroed after.
    on_step(data) is called after every step (the task-draw check records the object there)."""
    for _ in range(int(round(seconds / model.opt.timestep))):
        data.qpos[qa:qa + 7] = q0[:7]
        data.qpos[qadr] = q0[7:]
        data.qvel[da:da + 6] = 0
        data.qvel[dadr] = 0
        mujoco.mj_step(model, data)
        if on_step is not None:
            on_step(data)
    data.qpos[qa:qa + 7] = q0[:7]
    data.qpos[qadr] = q0[7:]
    data.qvel[:] = 0
    mujoco.mj_forward(model, data)


def tilt_deg(quat: np.ndarray, quat_ref: np.ndarray | None = None) -> np.ndarray:
    """Angle (deg) between the object's up axis and world vertical, or its up axis in quat_ref (the settled pose)."""
    up = qmat(np.asarray(quat))[..., :, 2]
    ref = np.array([0.0, 0.0, 1.0]) if quat_ref is None else qmat(np.asarray(quat_ref))[:, 2]
    return np.degrees(np.arccos(np.clip(up @ ref, -1, 1)))


def run_episode(task: dict, planner, C: dict, seed: int, out_dir: str, tk) -> dict:
    ec = C["episode"]
    model, spec, _, info = tk.compose_task(task)
    data = mujoco.MjData(model)
    q0, qa, da, qadr, dadr = place_robot(model, data, task)
    tr = SonicTracker(model, None)
    ob = model.body(info["object_body"]).id
    hb = model.body(HEAD_BODY).id
    spawn_pos, spawn_quat = data.xpos[ob].copy(), data.xquat[ob].copy()
    settle_s = float(ec.get("settle_s", SETTLE_S))
    settle_held(model, data, q0, qa, da, qadr, dadr, settle_s)
    obj0, quat0 = data.xpos[ob].copy(), data.xquat[ob].copy()   # episode reference: the settled object
    tr.reset_stream(data)
    if C["ego"].get("calibrated"):     # = the training renders (datagen): calibrated OAK-1, K-B fisheye 1280x800
        from humanoid_datagen.utils.oak_cam import OakCam
        r = OakCam(model, C["ego"]["camera"])
        render_ego = r.render
    else:
        cam = model.camera(C["ego"]["camera"]).id
        r = mujoco.Renderer(model, int(C["ego"]["height"]), int(C["ego"]["width"]))

        def render_ego(d):
            r.update_scene(d, camera=cam)
            return r.render().copy()
    planner.reset(task["instruction"], seed)
    dt = 0.02
    n = int(round(float(ec["timeout_s"]) / dt))
    every = int(round(float(ec["replan_s"]) / dt))
    active = MotionChunkCommand(np.concatenate([data.qpos[qa:qa + 7], data.qpos[tr.qadr]])[None], t0=0.0, fps=30.0)
    rec = dict(qpos=[], obj_pos=[], obj_quat=[], t=[])
    chunks, fell, plan_s = [], False, 0.0
    frames = []
    for k in range(n):
        t = k * dt
        prop = np.concatenate([data.qpos[qa:qa + 7], data.qpos[tr.qadr]])
        if k % every == 0:
            img = render_ego(data)
            if C["ego"].get("save_video"):
                frames.append(img)
            a = planner.act(dict(replan=True, time=t, images=img[None], instruction=task["instruction"],
                                 proprio_qpos=prop, task=task))
            if a.get("chunk") is not None:
                active = a["chunk"]
                plan_s += float(a.get("plan_s") or 0.0)
                chunks.append(dict(t0=t, frames=active.frames.astype(np.float32)))
        tr.act_stream(data, active.frame_at, t)
        for _ in range(DECIM):
            data.qfrc_applied[tr.dadr] = tr.pd_torque(data)
            mujoco.mj_step(model, data)
        if data.xpos[hb, 2] < FALL_HEAD_Z:
            fell = True
        rec["qpos"].append(data.qpos.copy())
        rec["obj_pos"].append(data.xpos[ob].copy())
        rec["obj_quat"].append(data.xquat[ob].copy())
        rec["t"].append(t + dt)
        if fell and ec.get("terminate_on_fall", True):
            break
    r.close()
    d = np.array(info["direction"])
    disp = (np.array(rec["obj_pos"])[:, :2] - obj0[:2]) @ d
    tilt = tilt_deg(np.array(rec["obj_quat"]), quat0)          # from the settled pose (success)
    tilt_abs = tilt_deg(np.array(rec["obj_quat"]))               # from vertical (the v2 rule; kept for rescoring)
    sc = tk.pass_rule                                   # the task's pass rule (tilt: from the settled pose)
    res = dict(task_id=task["task_id"], scene=task["scene"], object=task["object"]["name"], seed=seed,
               source=task.get("source"), type=task.get("type"), in_training_types=task.get("in_training_types"),
               push_dist=float(disp[-1]), push_dist_max=float(disp.max()), max_tilt_deg=float(tilt.max()),
               max_tilt_abs_deg=float(tilt_abs.max()), settle_s=settle_s,
               settle_move=float(np.linalg.norm(obj0 - spawn_pos)), settle_tilt_deg=float(tilt_deg(quat0, spawn_quat)),
               fell=fell,
               episode_s=float(rec["t"][-1]), n_plans=len(chunks), plan_ms_mean=1e3 * plan_s / max(len(chunks), 1))
    res["success"] = bool(res["push_dist"] >= sc["min_push"] and res["max_tilt_deg"] <= sc["max_tilt_deg"] and not fell)
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.join(out_dir, f"{task['task_id']}_s{seed}")
    np.savez_compressed(stem + ".npz", obj0=obj0, obj_quat0=quat0, spawn_pos=spawn_pos, spawn_quat=spawn_quat, direction=d, **{k: np.array(v) for k, v in rec.items()},
                        chunk_t0=np.array([c["t0"] for c in chunks]),
                        chunk_frames=np.array([c["frames"] for c in chunks], dtype=object), allow_pickle=True)
    if frames:
        import imageio.v2 as imageio
        imageio.mimsave(stem + "_ego.mp4", frames, fps=max(1, int(round(1 / float(ec["replan_s"])))))
    res["trajectory"] = stem + ".npz"
    return res


def run(C: dict, tk) -> None:
    """Closed-loop episodes of the config's planner on the task's GRScenes test tasks (eval mode `run`)."""
    tasks = load_tasks(C)
    if C.get("task_ids"):            # a subset of the fixed task list (previews); null = all tasks
        keep = list(C["task_ids"])
        tasks = sorted((t for t in tasks if t["task_id"] in keep), key=lambda t: keep.index(t["task_id"]))
    from humanoid_datagen.utils.config import build as build_target
    pc = C["planner"]
    planner = build_target(pc)                          # e.g. vla.halo.HaloPlanner(hcfg=...), or HoldPlanner
    root = repo_path(C["results_root"])
    os.makedirs(root, exist_ok=True)
    rec_path = os.path.join(root, "episodes.jsonl")
    done = {(json.loads(l)["task_id"], json.loads(l)["seed"]) for l in open(rec_path)} if os.path.exists(rec_path) else set()
    t0 = time.time()
    for seed in C["seeds"]:
        for task in tasks:
            if (task["task_id"], seed) in done:
                continue
            res = run_episode(task, planner, C, int(seed), os.path.join(root, "trajectories"), tk)
            with open(rec_path, "a") as f:
                f.write(json.dumps(res) + "\n")
            print(f"[push] {res['task_id']} s{seed}: success={res['success']} push={res['push_dist']:.2f}m "
                  f"tilt={res['max_tilt_deg']:.0f} fell={res['fell']} plans={res['n_plans']} [{time.time() - t0:.0f}s]",
                  flush=True)
    recs = [json.loads(l) for l in open(rec_path)]
    from humanoid_datagen.utils.stats import cluster_bootstrap, wilson_interval
    k = sum(r["success"] for r in recs)
    by_scene = {}
    for r in recs:
        by_scene.setdefault(r["scene"], []).append(float(r["success"]))
    def split(key):
        groups = {}
        for r in recs:
            groups.setdefault(str(r.get(key)), []).append(r)
        return {g: dict(n=len(v), success_rate=sum(r["success"] for r in v) / len(v),
                        wilson95=wilson_interval(sum(r["success"] for r in v), len(v)))
                for g, v in sorted(groups.items())}

    summ = dict(n=len(recs), success_rate=k / max(len(recs), 1), wilson95=wilson_interval(k, len(recs)),
                scene_bootstrap95=cluster_bootstrap(list(by_scene.values())) if len(by_scene) > 1 else None,
                push_dist_mean=float(np.mean([r["push_dist"] for r in recs])),
                fell_rate=float(np.mean([r["fell"] for r in recs])), planner=pc["_target_"].rsplit(".", 1)[1],
                checkpoint=(pc.get("hcfg") or {}).get("checkpoint"), success_rule=tk.pass_rule,
                by_source=split("source"), by_training_type=split("in_training_types"), by_type=split("type"))
    json.dump(summ, open(os.path.join(root, "summary.json"), "w"), indent=1)
    print(summ)


class GRScenesEval:
    """Closed-loop push / task eval in held-out GRScenes homes. A run config instantiates it:
        evaluator: {_target_: humanoid_datagen.evaluators.grscenes.GRScenesEval, planner: {_target_: ...}, ...}
    and scripts/eval.py calls the config's `mode`: build (draw the test set once), run, or viz."""

    def __init__(self, task, **cfg):
        self.task, self.cfg = task, cfg

    def build(self) -> None:
        build(self.cfg, self.task)

    def run(self) -> None:
        run(self.cfg, self.task)

    def viz(self) -> None:
        from humanoid_datagen.evaluators.viz import run as viz_run
        viz_run(self.cfg, self.task)

    def rescore(self) -> None:
        self.task.rescore(self.cfg)
