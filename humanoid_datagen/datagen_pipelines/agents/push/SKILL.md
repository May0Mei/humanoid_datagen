---
name: push-scene
description: Rebuild the room of a generated video as a MuJoCo scene with a physically simulated object that a Unitree G1 pushes, using a given set of G1 push motions; author the room, assets and object yourself, stage the push, check it against the video and in physics, and finalize. Use for "/push-scene out=<run dir>".
argument-hint: out=<run dir>
---

# Build a push scene from a video

Arguments: `$ARGUMENTS` (`out=<run dir>`, it is also your working directory; `prior=text` when there is no video).

**Text prior (`prior=text`):** there is no `prior/video.mp4`, and `keyframes` is unavailable. `prior/wan.txt` is
your scene brief, so build the room it describes. Everything else (deliverables, tools, checks) is the same.
Compare your renders against the brief instead of video stills.

## What you get

```
prior/video.mp4      a generated video: a person pushes an object across a room. Your scene reference.
prior/wan.txt        the prompt that produced the video
prior/kimodo.txt     the prompt that produced the robot motions
motions/*.csv        candidate Unitree G1 motions of the push (30 fps: root xyz, root quat wxyz, 29 joint angles)
```

## What you deliver

A simulation the provided tools can run, in the run dir:

- **`room.xml`**: a MuJoCo MJCF of the room with **no robot**. The floor is at z = 0 and the room should span
  enough floor for the push. It may contain anything: walls, furniture, your own meshes (OBJ/STL) and textures
  (PNG), materials, lights and cameras. It must have **exactly one body named `push_object` with a `<freejoint/>`**,
  which is the object being pushed. The tools place that body, so its pose in the file doesn't matter.
  - Give it a realistic mass and friction (object-floor; for wheels, an effective rolling resistance), estimated
    from what the object is, its size and material, and how it moves in the video.
  - Keep its colliding geoms simple: boxes, cylinders, capsules or convex meshes.
  - Static furniture should collide (`contype`/`conaffinity` ≠ 0). Purely decorative things, like wall art and
    foliage, can be visual only (`contype="0" conaffinity="0"`).
  - Add at least one fixed camera that frames the push the way the video does.
- **`task.json`**:
  ```json
  {"motion": "motions/<name>.csv", "start": [x, y], "yaw_deg": 0,
   "object_body": "push_object", "object_pose": "auto", "instruction": "Push the <object> toward the <landmark>."}
  ```
  The chosen motion is rotated so its push runs along `yaw_deg`, and translated so the robot starts at
  `start`. With `"object_pose": "auto"` the object is put where the palms meet it. You may give
  `[x, y, yaw_deg]` instead.
- **`assessment.json`**: your own honest review:
  ```json
  {"video_match": "...", "deviations": ["..."], "physics": "...", "motion_choice": "...", "remaining_issues": ["..."]}
  ```

## You are free to

- Write and run your own Python code and tools in the run dir.
  - `python` in this environment has mujoco, numpy, scipy, trimesh, coacd (convex decomposition), PIL,
    imageio, and ffmpeg on the PATH.
  - Generate meshes procedurally, decompose them into convex parts for collision, make textures from video
    stills (crop, rectify, tile) or procedurally, and write the MJCF by hand or with `mujoco.MjSpec`.
  - Use assets from the library (below), as they are, rescaled, or as parts of your own, for furniture and
    for the pushed object; mix them freely with what you build yourself.
- Choose the room size, the furniture and its placement, the motion among the candidates, the start pose,
  the push direction, and the object's exact size, mass and friction.
- Iterate as often as useful, within the session budget.

- Install extra Python packages into the run dir (e.g. `pip install --target ./pylib <pkg>` and put `./pylib` on
  `PYTHONPATH`; the shared environment is read-only), and use the web, e.g. for reference images, textures or
  free 3D assets.
- Read the tools' source in `/opt/push_tools/` if you want to know exactly what they do.

## Provided tools

Run them from the run dir as `python -m humanoid_datagen.datagen_pipelines.agents.cli <tool>`. Each prints one JSON object.

| tool | what it does | time |
|---|---|---|
| `keyframes [--video prior/video.mp4] [--n 12]` | stills + contact sheet in `prior_frames/` | ~5 s |
| `motions` | per motion: time and **palm height** at first contact, how far the hands travel, whether the robot can perform it on its own; side-view sheet `motions/<m>_sheet.png` | ~20 s |
| `compose` | builds G1 + room + staged object; static checks (palms on the object's face, start pose collision-free) | ~3 s |
| `render [--frames start,contact,end] [--cameras a,b]` | renders your cameras, `oak1` (robot head camera) and `top` → `renders/rNN/sheet.png` | ~5 s |
| `simulate` | settles the object, then a learned whole-body tracker plays the motion with full physics; also a kinematic replay for comparison → `sims/sNN/summary.json`, `track_sheet.png`, videos | ~1 min |
| `finalize --assessment assessment.json` | final simulate, copies the deliverables to `final/`, writes `result.json` | ~1 min |
| `room_from_layout --layout L.json` | optional: a quick primitive-only room from a JSON layout (run it without `--layout` to get the format) | ~1 s |

**Asset library** (`python -m humanoid_datagen.datagen_pipelines.shared.library ...`): ~7.6k metric-scale 3D assets (AI2-THOR artist
furniture and objects, Objaverse home objects), each with textured visual meshes and convex colliders.

| command | what it does | time |
|---|---|---|
| `search "<description>" [--k 8] [--category c]` | CLIP text search; uid, category, description, size, mass, license per hit, thumbnail sheet `library/search_NN.png` | ~20 s |
| `export <uid> --name N [--height h \| --scale s] [--static]` | `library/N/asset.xml`: one body `N`, origin at its bottom centre, z up, front +x, mass on the colliders; attach it with `MjSpec` (see the module docstring) | ~3 s |
| `categories` | category names with asset counts | ~1 s |

A simulate result passes when the static checks pass, the robot stays upright, the object moves at least 0.3 m
along the push direction, and it tilts less than 20°.

## Procedure

1. **Read the prior.** Run `keyframes`, Read `prior_frames/sheet.png` and 2–3 stills, and Read the three text
   files. Write `notes.md` with:
   - room type and size;
   - wall, floor and furniture list with rough positions;
   - colours and materials;
   - the object: look, size from scale cues (doors ≈ 2.0 m, seats ≈ 0.45 m, counters ≈ 0.9 m), and plausible
     mass and friction;
   - where the push starts and where it goes;
   - what the video doesn't show, which you fill to make a complete, plausible room.
2. **Pick a motion.** Run `motions` and Read the sheets.
   - The object must be taller than the motion's palm height, or the palms miss it. A low crate needs a
     crouching motion.
   - The push distance must fit the free floor in your room.
3. **Build the room and object.** Write `room.xml` (and any assets and code), then `task.json`. Run `compose`
   and fix any failing check.
4. **Look.** Run `render` and Read the sheet next to the video stills:
   - Does the room match: shell, colours, furniture placement, light?
   - Does the object look and scale like the video's?
   - Does `oak1` look into the room at the object?
   - Does `top` show a clear path for the robot and the object?
5. **Simulate.** Run `simulate` and Read `track_sheet.png` and `summary.json`. If it fails, fix the cause:
   - Object tips over: make it deeper or lower, or lower its friction.
   - Robot stalls or falls: object too heavy or too sticky, or an obstacle in the path.
   - Object never touched: wrong motion for its height, or staging off.

   Then compare the tracked push with the kinematic replay. A large gap is fine if the physics is plausible.
6. **Iterate** on the worst mismatch, against the video or the physics, until the scene is faithful and
   `simulate` passes. Always look at the pictures; the worst bugs show only there.
7. **Finalize.** Write `assessment.json` and run `finalize --assessment assessment.json`. Always finalize,
   even if something still fails: a run without `result.json` counts as a failure. Reply with one line giving
   the status, the motion used, the push distance, and the main remaining issue.
