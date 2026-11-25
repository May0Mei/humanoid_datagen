"""Push placement: the object against the staged push motion's palms, or ahead of the robot without a motion; and
the push scene's static checks (moved out of datagen_pipelines/shared/compose.py)."""
from __future__ import annotations

import numpy as np

from humanoid_datagen.datagen_pipelines.shared.compose import robot_only_model, start_pose_check
from humanoid_datagen.tasks.push.stage import stage
from humanoid_datagen.utils.sonic import Reference


def stage_motion(ref0: Reference, ext: dict, start, yaw_deg: float):
    """Rotate / translate the motion so its push runs along push_yaw from `start`; object pose with its AABB's near
    face on the palms at the contact frame, centred on the hands.
    Returns (ref, pose [x, y, yaw_deg], st)."""
    st = stage(robot_only_model(), ref0, ext["half"], friction=0.0, tip_safety=0.0)  # pure kinematics; sizes untouched
    yaw_push = np.radians(yaw_deg)
    rot = yaw_push - np.arctan2(st["push_dir"][1], st["push_dir"][0])
    c, s = np.cos(rot), np.sin(rot)
    R2 = np.array([[c, -s], [s, c]])
    start = np.asarray(start, float)
    ref = ref0.transformed(start, rot)
    d = np.array([np.cos(yaw_push), np.sin(yaw_push)])
    hand_xy = R2 @ (np.array(st["box_center"]) - ref0.root_pos[0, :2] - np.array(st["push_dir"]) * st["box_half"][0]) + start
    cxy = hand_xy + d * ext["half"][0]
    Ro = np.array([[c_ := np.cos(yaw_push), -np.sin(yaw_push)], [np.sin(yaw_push), c_]])
    oxy = cxy - Ro @ np.array(ext["center"][:2])
    return ref, [float(oxy[0]), float(oxy[1]), float(yaw_deg)], st


def pose_at_distance(ext: dict, start, yaw_deg: float, dist: float, lateral: float = 0.0,
                     center_com: bool = False):
    """Object pose with its near face `dist` m ahead of the start pelvis along push_yaw (no motion needed).
    `lateral` offsets the box centre sideways; center_com=True puts the centre of mass (ext["com"]) on that line
    instead, so a straight push moves the object rather than turning it (distance to the near face unchanged)."""
    y = np.radians(yaw_deg)
    d, lat = np.array([np.cos(y), np.sin(y)]), np.array([-np.sin(y), np.cos(y)])
    cxy = np.asarray(start, float) + d * (dist + ext["half"][0]) + lat * lateral
    Ro = np.array([[np.cos(y), -np.sin(y)], [np.sin(y), np.cos(y)]])
    if center_com:
        cxy = cxy - lat * float(lat @ (Ro @ (np.array(ext["com"][:2]) - np.array(ext["center"][:2]))))
    oxy = cxy - Ro @ np.array(ext["center"][:2])
    return [float(oxy[0]), float(oxy[1]), float(yaw_deg)]


def checks(model, ref: Reference, info: dict) -> dict:
    """Static checks: palms below the object's top and above its bottom, robot start pose collision-free."""
    ext = info["object_extent"]
    top = ext["center"][2] + ext["half"][2] - ext["z_low"]
    hz = info["contact_height"]
    out = dict(palms_on_object_face=bool(0.05 < hz < top - 0.03), palm_height=round(hz, 3), object_top=round(top, 3))
    out.update(start_pose_check(model, ref, info))
    out["body_intrusion_m"] = round(info["body_intrusion"], 3)
    out["ok"] = bool(out["palms_on_object_face"] and out["start_pose_free"])
    return out
