"""The OAK-1 head camera as HALO renders it, at the 2026-11-19 calibration (assets/robots/g1/oak1_calibration.json).

Pose: oak1 on helmet_assembly_link at the calibration's mujoco_camera_in_helmet_assembly_link (pos, xyaxes).
Image: HALO's recipe (ml-halo/datagen/render): a 140-deg-HFOV pinhole render (load_oak_intrinsics' pinhole_render,
1600x1000), warped with cv2.remap to the calibrated Kannala-Brandt fisheye, 1280x800. Shared by the training datagen
(datagen) and the eval (evaluators/grscenes), so both see the same camera.

    cam = OakCam(model)            # edits model's oak1 camera in place
    img = cam.render(data)         # (800, 1280, 3) uint8 fisheye
"""
from __future__ import annotations

import json
import os
import sys

import mujoco
import numpy as np

from humanoid_datagen.utils.paths import repo_path, third_party

CALIB = repo_path("humanoid_datagen/assets/robots/g1/oak1_calibration.json")
_INTR = None


def intrinsics() -> dict:
    global _INTR
    if _INTR is None:
        p = third_party("ml-halo", "datagen")
        if p not in sys.path:
            sys.path.insert(0, p)
        from render.cameras import load_oak_intrinsics  # HALO
        _INTR = load_oak_intrinsics(CALIB)
    return _INTR


def calibrate(model: mujoco.MjModel, name: str = "oak1") -> int:
    """Set the camera's pose to the calibration and its fovy to HALO's pinhole render; returns the camera id."""
    pr = intrinsics()["pinhole_render"]
    cal = json.load(open(CALIB))["extrinsics"]["mujoco_camera_in_helmet_assembly_link"]
    c = model.camera(name).id
    assert model.body(model.cam_bodyid[c]).name == "helmet_assembly_link"
    x, y = np.array(cal["xyaxes"].split(), float).reshape(2, 3)
    x, y = x / np.linalg.norm(x), y / np.linalg.norm(y)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.stack([x, y, np.cross(x, y)], 1).ravel())
    model.cam_pos[c] = np.array(cal["pos"].split(), float)
    model.cam_quat[c] = q
    model.cam_sensorsize[c] = 0          # use fovy, not the sensor intrinsics
    model.cam_fovy[c] = np.degrees(2 * np.arctan(pr["height"] / 2 / pr["fy_px"]))
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, int(pr["width"]))
    model.vis.global_.offheight = max(model.vis.global_.offheight, int(pr["height"]))
    return c


class OakCam:
    def __init__(self, model: mujoco.MjModel, name: str = "oak1"):
        I = intrinsics()                                       # puts ml-halo/datagen on sys.path
        from render.fisheye_warp import _build_fisheye_remap  # HALO
        self.cam = calibrate(model, name)
        pr = I["pinhole_render"]
        self.r = mujoco.Renderer(model, int(pr["height"]), int(pr["width"]))
        self.mx, self.my = _build_fisheye_remap(I)
        self.width, self.height = int(I["width"]), int(I["height"])

    def render(self, data: mujoco.MjData) -> np.ndarray:
        import cv2
        self.r.update_scene(data, camera=self.cam)
        return cv2.remap(self.r.render(), self.mx, self.my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)

    def close(self):
        self.r.close()
