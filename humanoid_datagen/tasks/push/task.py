"""The push task: push a household object forward with the hands. Instantiated from a run config:

    task:
      _target_: humanoid_datagen.tasks.push.task.PushTask
      pass_rule: {min_push: 0.3, max_tilt_deg: 20.0}
      hand_rule: {min_hand_m: 0.3, max_leg_body_m: 0.1, min_both_frac: 0.0}   # null: no hand-contact filter
      motion_fit: {min_top_clear: 0.05, min_palm_frac: 0.35, min_push: 1.0}
      approach_prompt: "a person walks up to {a} {object} and bends over to push it."

The parts: stage.py (kinematic staging of a push motion against an object), placement.py (the object in a scene),
spots.py (robot starts with a free push corridor), reward.py (metrics, hand-contact attribution), prompts.py (prior
prompts), test_objects.py (GRScenes test set), hand_rescore.py. Heavy parts import lazily, so the agent sandbox only
needs base.py, this file, stage.py, placement.py and reward.py.
"""
from __future__ import annotations

import numpy as np

from humanoid_datagen.tasks.base import Task


class PushTask(Task):
    name = "push"
    halo_gate = "push_box"

    def __init__(self, pass_rule: dict, hand_rule: dict | None = None, motion_fit: dict | None = None,
                 approach_prompt: str = "a person walks up to {a} {object} and bends over to push it.",
                 corridor_half_width: tuple = (0.15, 0.35)):
        super().__init__(pass_rule=dict(pass_rule), hand_rule=dict(hand_rule) if hand_rule else None,
                         motion_fit=dict(motion_fit or {}), approach_prompt=approach_prompt,
                         corridor_half_width=list(corridor_half_width))
        self.pass_rule = dict(pass_rule)
        self.hand_rule = dict(hand_rule) if hand_rule else None
        self.fit = dict(motion_fit or {})
        self._approach = approach_prompt
        self._hw = tuple(corridor_half_width)          # corridor half width: object half width + [0], at least [1]

    # --- language
    def instruction(self, object_type: str) -> str:
        return f"Push the {object_type.replace('_', ' ')} forward."

    def approach_prompt(self, object_type: str) -> str:
        obj = object_type.replace("_", " ")
        return self._approach.format(a="an" if obj[0] in "aeiou" else "a", object=obj)

    # --- motions
    def stage(self, robot_model, ref, half_extents, **kw) -> dict:
        from humanoid_datagen.tasks.push.stage import stage
        return stage(robot_model, ref, half_extents, **kw)

    def motion_fit(self, robot_model, ref, object_height: float) -> dict:
        """Palms meet the object's near face >= min_top_clear below its top and at >= min_palm_frac of its height;
        candidate_ok also needs the hands to travel >= min_push."""
        st = self.stage(robot_model, ref, [0.1, 0.1, 0.1], friction=0.0, tip_safety=0.0)
        palm, push = float(st["contact_height"]), float(st["kinematic_push_dist"])
        H = float(object_height)
        fits = H - palm >= self.fit["min_top_clear"] and palm >= self.fit["min_palm_frac"] * H
        return dict(palm_height=round(palm, 3), push=round(push, 2), fits=fits,
                    candidate_ok=bool(fits and push >= self.fit["min_push"]))

    def motion_frame(self, robot_model, ref, half_extents) -> dict:
        """The push direction, the reach from frame 0's pelvis to the object's far face, and the kinematic push."""
        st = self.stage(robot_model, ref, half_extents, friction=0.0, tip_safety=0.0)
        u = np.asarray(st["push_dir"], float)
        near = float((np.asarray(st["box_center"]) - ref.root_pos[0, :2]) @ u - st["box_half"][0])
        return dict(dir=u, reach=near + 2 * float(half_extents[0]), travel=float(st["kinematic_push_dist"]))

    # --- scenes
    def place_object(self, ref0, ext: dict, start, yaw_deg: float):
        from humanoid_datagen.tasks.push.placement import stage_motion
        ref, pose, st = stage_motion(ref0, ext, start, yaw_deg)
        staged = dict(contact_frame=st["frame_contact"], t_contact=st["t_contact"], contact_height=st["contact_height"],
                      kinematic_push_dist=st["kinematic_push_dist"], body_intrusion=st["body_intrusion"])
        return ref, pose, staged

    def object_pose(self, ext: dict, start, yaw_deg: float, dist: float, lateral: float = 0.0, exact: bool = False):
        from humanoid_datagen.tasks.push.placement import pose_at_distance
        return pose_at_distance(ext, start, yaw_deg, dist, lateral, center_com=exact)

    def scene_checks(self, model, ref, info: dict) -> dict:
        from humanoid_datagen.tasks.push.placement import checks
        return checks(model, ref, info)

    def spot_ok(self, model, data, xy, yaw: float, ext: dict, ahead: float, back: float, exclude_body: int = -1) -> bool:
        from humanoid_datagen.datagen_pipelines.shared.sampler import _floor_ok
        from humanoid_datagen.tasks.push.spots import corridor_ok
        hw = max(float(ext["half"][1]) + self._hw[0], self._hw[1])
        return bool(_floor_ok(model, data, xy, 0.04, exclude_body)
                    and corridor_ok(model, data, xy, yaw, ahead, back, hw, exclude_body=exclude_body))

    # --- success
    def metrics(self, traj: dict, ref, direction, model) -> dict:
        from humanoid_datagen.tasks.push.reward import metrics
        return metrics(traj, ref, direction, model)

    def passed(self, m: dict) -> bool:
        return bool((not m["robot_fell"]) and m["push_dist"] >= self.pass_rule["min_push"]
                    and m["max_tilt_deg"] <= self.pass_rule["max_tilt_deg"])

    def accept_demo(self, model, traj: dict, info: dict) -> dict:
        """The hand rule: the hands, not the legs, moved the object (reward.py attribution)."""
        if not self.hand_rule:
            return {"pass": True}
        from humanoid_datagen.datagen_pipelines.shared.compose import ROBOT_CONTYPE
        from humanoid_datagen.tasks.push.reward import attribute, attribution_model, passes
        m, obj, gp = attribution_model(model, info["object_body"], ROBOT_CONTYPE)
        hc = attribute(m, obj, gp, np.asarray(traj["qpos"]), np.asarray(traj["obj_pos"]), info["direction"])
        hc["pass"] = passes(hc, self.hand_rule)
        return hc

    # --- data and test set
    def write_priors(self, cfg: dict) -> None:
        from humanoid_datagen.tasks.push.prompts import write
        write(cfg)

    def build_tasks(self, cfg: dict) -> list[dict]:
        from humanoid_datagen.tasks.push.test_objects import build_tasks
        return build_tasks(cfg, self)

    def compose_task(self, task: dict, **kw):
        from humanoid_datagen.tasks.push.test_objects import compose_task
        return compose_task(task, self, **kw)

    def rescore(self, cfg: dict) -> None:
        from humanoid_datagen.tasks.push.hand_rescore import run
        run(cfg, self)
