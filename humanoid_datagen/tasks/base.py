"""The task interface: everything task-specific that the shared pipeline code, the agent tools and the evaluators
ask of a task. A run config instantiates one (`task: {_target_: humanoid_datagen.tasks.<name>.task.<Name>Task, ...}`)
and passes it to the stage / evaluator; shared code never imports a task package itself."""
from __future__ import annotations


class Task:
    name = "task"

    def __init__(self, **params):
        self.params = params            # the constructor arguments, so to_config() can rebuild the task

    def to_config(self) -> dict:
        """The Hydra config that rebuilds this task (the agent tools get it through $HDG_TASK)."""
        return {"_target_": f"{type(self).__module__}.{type(self).__qualname__}", **self.params}

    # --- language
    def instruction(self, object_type: str) -> str:
        raise NotImplementedError

    # --- motions (motion selection, lead-ins, datagen)
    def motion_fit(self, robot_model, ref, object_height: float) -> dict:
        """Kinematic fit of a candidate motion to the object: {fits, keep_candidate, ...numbers}."""
        raise NotImplementedError

    def motion_frame(self, robot_model, ref, half_extents) -> dict:
        """The motion's action frame: {dir (unit 2-d), reach (m ahead of frame 0 to the object's far face), travel (m)}."""
        raise NotImplementedError

    def approach_prompt(self, object_type: str) -> str:
        """Kimodo prompt of the walk-up into the task motion."""
        raise NotImplementedError

    halo_gate = None                    # HALO motion_check task type that ranks walk-up samples

    # --- scenes (compose, datagen spots)
    def place_object(self, ref0, ext: dict, start, yaw_deg: float):
        """Stage a motion at `start` facing yaw_deg and put the object where the motion acts:
        (ref, object pose [x, y, yaw_deg], staging info dict)."""
        raise NotImplementedError

    def object_pose(self, ext: dict, start, yaw_deg: float, dist: float, lateral: float = 0.0, exact: bool = False):
        """Object pose without a motion (eval tasks): `dist` m ahead of the start."""
        raise NotImplementedError

    def scene_checks(self, model, ref, info: dict) -> dict:
        """Static checks of a composed scene: {ok, ...}."""
        raise NotImplementedError

    def spot_ok(self, model, data, xy, yaw: float, ext: dict, ahead: float, back: float, exclude_body: int = -1) -> bool:
        """Is a robot start (xy, yaw) usable in this room for the task (free floor / corridor)?"""
        raise NotImplementedError

    # --- success
    def metrics(self, traj: dict, ref, direction, model) -> dict:
        raise NotImplementedError

    def passed(self, metrics: dict) -> bool:
        raise NotImplementedError

    def accept_demo(self, model, traj: dict, info: dict) -> dict:
        """Extra demo acceptance beyond `passed`: {pass, ...record}."""
        return {"pass": True}

    # --- data and test set
    def write_priors(self, cfg: dict) -> None:
        raise NotImplementedError

    def build_tasks(self, cfg: dict) -> list[dict]:
        raise NotImplementedError

    def compose_task(self, task: dict, **kw):
        raise NotImplementedError

    def rescore(self, cfg: dict) -> None:
        raise NotImplementedError
