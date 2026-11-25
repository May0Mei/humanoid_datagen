"""GEAR-SONIC motion tracker (NVlabs GR00T-WholeBodyControl) driven from Python in MuJoCo.

Re-implements the observation pipeline of the released C++ deploy stack
(gear_sonic_deploy/src/g1/g1_deploy_onnx_ref, observation_config.yaml, encoder mode "g1")
so the released ONNX encoder/decoder can track a G1 joint-space reference at 50 Hz.

Conventions (from policy_parameters.hpp):
  * MuJoCo joint order = our G1 MJCF / Kimodo CSV order. The policy uses IsaacLab order.
    `MJ_OF_IL[i]` is the MuJoCo index of IsaacLab joint i (the C++ `mujoco_to_isaaclab`),
    `IL_OF_MJ[j]` is the IsaacLab index of MuJoCo joint j (the C++ `isaaclab_to_mujoco`).
  * Quaternions are wxyz. Heading = normalize(w, 0, 0, z).
  * Decoder input (994) = token 64 | ang vel 10x3 | joint pos-default 10x29 | joint vel 10x29
    | last action 10x29 | gravity 10x3, histories oldest first, all joint data IsaacLab order.
  * Encoder input (1762), mode g1: [mode 4 | ref joint pos 10x29 (step 5) | ref joint vel 10x29
    | ... zeros ... | anchor orientation 10x6 at offset 601 | zeros].
  * target_mj[j] = default[j] + action[IL_OF_MJ[j]] * action_scale[j]; PD with SONIC kp/kd.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np

CKPT_DIR = os.environ.get("GEAR_SONIC_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                                       "utilities", "checkpoints", "gear_sonic"))

IL_OF_MJ = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
                     11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28])
MJ_OF_IL = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10,
                     16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28])
assert (IL_OF_MJ[MJ_OF_IL] == np.arange(29)).all()

JOINT_NAMES = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
]

# motor constants (policy_parameters.hpp)
_W = 10 * 2.0 * np.pi
_ARM = {"5020": 0.003609725, "7520_14": 0.010177520, "7520_22": 0.025101925, "4010": 0.00425}
_EFF = {"5020": 25.0, "7520_14": 88.0, "7520_22": 139.0, "4010": 5.0}
_LEG = ["7520_22", "7520_22", "7520_14", "7520_22", "5020", "5020"]
_ARMJ = ["5020", "5020", "5020", "5020", "5020", "4010", "4010"]
MOTOR = _LEG + _LEG + ["7520_14", "5020", "5020"] + _ARMJ + _ARMJ
_DOUBLE = {4, 5, 10, 11, 13, 14}  # ankles, waist roll/pitch use 2x stiffness/damping
KP = np.array([_ARM[m] * _W ** 2 * (2.0 if j in _DOUBLE else 1.0) for j, m in enumerate(MOTOR)])
KD = np.array([2.0 * 2.0 * _ARM[m] * _W * (2.0 if j in _DOUBLE else 1.0) for j, m in enumerate(MOTOR)])
ACTION_SCALE = np.array([0.25 * _EFF[m] / (_ARM[m] * _W ** 2) for m in MOTOR])
DEFAULT = np.array([-0.312, 0, 0, 0.669, -0.363, 0, -0.312, 0, 0, 0.669, -0.363, 0, 0, 0, 0,
                    0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0])
CONTROL_DT = 0.02


# ---------------------------------------------------------------- quaternion helpers (wxyz)
def qmul(a, b):
    w1, x1, y1, z1 = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    w2, x2, y2, z2 = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                     w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                     w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2], -1)


def qconj(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def qrot(q, v):
    qv = np.concatenate([np.zeros(v.shape[:-1] + (1,)), v], -1)
    return qmul(qmul(q, qv), qconj(q))[..., 1:]


def heading(q):
    h = np.zeros_like(q)
    h[..., 0], h[..., 3] = q[..., 0], q[..., 3]
    return h / np.maximum(np.linalg.norm(h, axis=-1, keepdims=True), 1e-9)


def qmat(q):
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def slerp(q0, q1, t):
    d = np.sum(q0 * q1, -1, keepdims=True)
    q1 = np.where(d < 0, -q1, q1)
    d = np.abs(d)
    th = np.arccos(np.clip(d, -1, 1))
    s = np.sin(th)
    small = s < 1e-6
    a = np.where(small, 1 - t, np.sin((1 - t) * th) / np.where(small, 1, s))
    b = np.where(small, t, np.sin(t * th) / np.where(small, 1, s))
    q = a * q0 + b * q1
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


# ---------------------------------------------------------------- reference motion
@dataclass
class Reference:
    """50 Hz reference. root_pos (T,3), root_quat (T,4) wxyz, joint_pos/vel (T,29) MuJoCo order."""
    root_pos: np.ndarray
    root_quat: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray

    @property
    def T(self):
        return len(self.joint_pos)

    @classmethod
    def from_qpos(cls, q: np.ndarray, fps: float, out_fps: float = 50.0) -> "Reference":
        """q: (N, 36) = root xyz, root quat wxyz, 29 joints (Kimodo output.csv layout)."""
        n = len(q)
        t_src = np.arange(n) / fps
        t = np.arange(0, t_src[-1] + 1e-9, 1.0 / out_fps)
        u = np.clip(t * fps, 0, n - 1)
        i0 = np.floor(u).astype(int)
        i1 = np.minimum(i0 + 1, n - 1)
        a = (u - i0)[:, None]
        pos = q[i0, :3] * (1 - a) + q[i1, :3] * a
        quat = slerp(q[i0, 3:7], q[i1, 3:7], a)
        jp = q[i0, 7:] * (1 - a) + q[i1, 7:] * a
        jv = np.gradient(jp, 1.0 / out_fps, axis=0)
        return cls(pos, quat, jp, jv)

    def transformed(self, xy, yaw) -> "Reference":
        """Rigidly move the motion: rotate by yaw about z (around the frame-0 root), then put frame-0 root at xy."""
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        p0 = self.root_pos[0] * np.array([1, 1, 0])
        pos = (self.root_pos - p0) @ R.T + np.array([xy[0], xy[1], 0.0])
        qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        quat = qmul(np.broadcast_to(qz, self.root_quat.shape), self.root_quat)
        return Reference(pos, quat, self.joint_pos.copy(), self.joint_vel.copy())


# ---------------------------------------------------------------- tracker
class SonicTracker:
    """Call `reset(data)` once with the robot in the frame-0 pose, then `act(data)` every 20 ms.

    `act` returns the PD joint targets (MuJoCo order); apply them with `pd_torque` at the sim rate.
    """

    def __init__(self, model: mujoco.MjModel, ref: Reference, ckpt_dir: str = CKPT_DIR, free_joint: str | None = None):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = 2
        self.enc = ort.InferenceSession(os.path.join(ckpt_dir, "model_encoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.dec = ort.InferenceSession(os.path.join(ckpt_dir, "model_decoder.onnx"), so, providers=["CPUExecutionProvider"])
        self.ref = ref
        self.qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
        self.dadr = np.array([model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES])
        fj = model.joint(free_joint) if free_joint else robot_free_joint(model)
        assert model.jnt_type[fj.id] == mujoco.mjtJoint.mjJNT_FREE
        self.base_q = model.jnt_qposadr[fj.id]
        self.base_d = model.jnt_dofadr[fj.id]
        self.frame = 0
        self.hist = []
        self.last_action = np.zeros(29)
        self.target = DEFAULT.copy()

    # robot state -------------------------------------------------------
    def _state(self, data):
        quat = data.qpos[self.base_q + 3:self.base_q + 7].copy()
        angvel = data.qvel[self.base_d + 3:self.base_d + 6].copy()  # free joint: local frame
        q = data.qpos[self.qadr]
        dq = data.qvel[self.dadr]
        return dict(quat=quat, angvel=angvel,
                    q=(q - DEFAULT)[MJ_OF_IL], dq=dq[MJ_OF_IL],
                    grav=qrot(qconj(quat), np.array([0.0, 0.0, -1.0])),
                    last_action=self.last_action.copy())

    def reset(self, data):
        self.frame = 0
        self.last_action = np.zeros(29)
        self.init_base_quat = data.qpos[self.base_q + 3:self.base_q + 7].copy()
        # apply_delta_heading = heading(robot at start) * heading(ref frame 0)^-1
        self.delta = qmul(heading(self.init_base_quat), qconj(heading(self.ref.root_quat[0])))
        s = self._state(data)
        self.hist = [s] * 10

    # observations ------------------------------------------------------
    def _future(self, step=5, n=10):
        return np.minimum(self.frame + step * np.arange(n), self.ref.T - 1)

    def encoder_obs(self, base_quat):
        idx = self._future()
        obs = np.zeros(1762)
        obs[4:294] = self.ref.joint_pos[idx][:, MJ_OF_IL].reshape(-1)
        obs[294:584] = self.ref.joint_vel[idx][:, MJ_OF_IL].reshape(-1)
        ref_q = qmul(np.broadcast_to(self.delta, (len(idx), 4)), self.ref.root_quat[idx])
        rel = qmul(np.broadcast_to(qconj(base_quat), ref_q.shape), ref_q)
        obs[601:661] = qmat(rel)[:, :, :2].reshape(-1)
        return obs

    def decoder_obs(self, token):
        h = self.hist
        return np.concatenate([token,
                               np.concatenate([e["angvel"] for e in h]),
                               np.concatenate([e["q"] for e in h]),
                               np.concatenate([e["dq"] for e in h]),
                               np.concatenate([e["last_action"] for e in h]),
                               np.concatenate([e["grav"] for e in h])])

    def act(self, data) -> np.ndarray:
        s = self._state(data)
        self.hist = self.hist[1:] + [s]
        e = self.encoder_obs(s["quat"]).astype(np.float32)[None]
        token = self.enc.run(None, {"obs_dict": e})[0][0]
        d = self.decoder_obs(token).astype(np.float32)[None]
        a = self.dec.run(None, {"obs_dict": d})[0][0].astype(np.float64)
        self.last_action = a
        self.target = DEFAULT + a[IL_OF_MJ] * ACTION_SCALE
        self.frame = min(self.frame + 1, self.ref.T - 1)
        return self.target

    # streamed references (planner chunks) ---------------------------------------------------------------
    def reset_stream(self, data):
        """Start tracking world-frame references given per tick by act_stream (no heading re-alignment)."""
        self.frame = 0
        self.last_action = np.zeros(29)
        self.delta = np.array([1.0, 0.0, 0.0, 0.0])
        self.hist = [self._state(data)] * 10

    def act_stream(self, data, frame_at, t: float) -> np.ndarray:
        """frame_at(t) -> world qpos (36: xyz, wxyz, 29 joints MuJoCo order), e.g. MotionChunkCommand.frame_at.
        The encoder sees frames at t + 0.1 k (k = 0..9), joint velocities by finite differences."""
        s = self._state(data)
        self.hist = self.hist[1:] + [s]
        ts = t + 0.1 * np.arange(10)
        F = np.stack([frame_at(x) for x in ts])
        F2 = np.stack([frame_at(x + CONTROL_DT) for x in ts])
        jp, jv = F[:, 7:], (F2[:, 7:] - F[:, 7:]) / CONTROL_DT
        obs = np.zeros(1762)
        obs[4:294] = jp[:, MJ_OF_IL].reshape(-1)
        obs[294:584] = jv[:, MJ_OF_IL].reshape(-1)
        q = F[:, 3:7] / np.linalg.norm(F[:, 3:7], axis=1, keepdims=True)
        rel = qmul(np.broadcast_to(qconj(s["quat"]), q.shape), q)
        obs[601:661] = qmat(rel)[:, :, :2].reshape(-1)
        token = self.enc.run(None, {"obs_dict": obs.astype(np.float32)[None]})[0][0]
        a = self.dec.run(None, {"obs_dict": self.decoder_obs(token).astype(np.float32)[None]})[0][0].astype(np.float64)
        self.last_action = a
        self.target = DEFAULT + a[IL_OF_MJ] * ACTION_SCALE
        return self.target

    def pd_torque(self, data, target=None):
        target = self.target if target is None else target
        tau = KP * (target - data.qpos[self.qadr]) - KD * data.qvel[self.dadr]
        lim = np.array([_EFF[m] for m in MOTOR])
        return np.clip(tau, -lim, lim)


def robot_free_joint(model):
    """The free joint of the G1 pelvis (other free bodies, e.g. the pushed object, may come first)."""
    pel = model.body("pelvis").id
    for j in range(model.njnt):
        if model.jnt_bodyid[j] == pel and model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
            return model.joint(j)
    raise ValueError("no free joint on body 'pelvis'")


def set_robot_pose(model, data, ref: Reference, i: int, free_joint: str | None = None, z_offset: float = 0.0):
    fj = model.joint(free_joint) if free_joint else robot_free_joint(model)
    a = model.jnt_qposadr[fj.id]
    data.qpos[a:a + 3] = ref.root_pos[i] + np.array([0, 0, z_offset])
    data.qpos[a + 3:a + 7] = ref.root_quat[i]
    qadr = np.array([model.jnt_qposadr[model.joint(n).id] for n in JOINT_NAMES])
    data.qpos[qadr] = ref.joint_pos[i]
    data.qvel[model.jnt_dofadr[fj.id]:model.jnt_dofadr[fj.id] + 6] = 0
    data.qvel[[model.jnt_dofadr[model.joint(n).id] for n in JOINT_NAMES]] = 0
    mujoco.mj_forward(model, data)
