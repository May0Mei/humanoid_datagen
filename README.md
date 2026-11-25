# Humanoid Data Generation

Push-task demos for the Unitree G1 from video priors: a Wan2.2 video of a person pushing a household object in a
home, Kimodo G1 push motions, a Claude agent that rebuilds the room as a MuJoCo scene, and GEAR-SONIC physics demos.
The demos train the HALO VLA, which is evaluated closed-loop in held-out GRScenes homes. Pipelines: the agent
(`prior: video`, ours, or `prior: text`) and the HALO-Infinigen baseline.

**Run every command from the repo root.** Every run is one config file, composed with Hydra. Its blocks with a
`_target_` are instantiated: `task` (the task), and `stage` (data generation), `evaluator` (eval) or `step` (training).
The task is handed to the stage / evaluator, which the entry script then runs:

```
python humanoid_datagen/scripts/datagen.py --config_name datagen/push/agents/demos      # build(task), build(stage, task).run()
python humanoid_datagen/scripts/train.py   --config_name train/push/halo_ours           # build(step).run()
python humanoid_datagen/scripts/eval.py    --config_name eval/push/grscenes_halo_ours   # build(evaluator, task).<mode>()
```

`--config_name X` reads `humanoid_datagen/configs/X.yaml`; paths in configs are relative to the repo root.

## Layout

```
humanoid_datagen/
  assets/                  robots/g1 (MJCFs, meshes, OAK-1 calibration), pushbox_tex
  configs/
    task/push.yaml         the push task (PushTask: pass rule, hand rule, motion fit, walk-up prompt), pulled into
                           every datagen / eval config (`- /task/push@task`)
    datagen/push/          agents/ (video prior), agents_text/ (text prior), halo_infinigen/: base.yaml + one config
                           per stage (a `stage` block); prompts_1000.jsonl
    train/push/            export, halo_<arm> (a `step` block); vla/halo/push_<arm>.yaml: HALO-format training configs
    eval/push/             grscenes_hold (base) + grscenes_halo_<arm>, build_grscenes, rescore, viz (an `evaluator`
                           block and its `mode`); test_split.json
  tasks/
    base.py                Task: the interface shared code calls (place_object, spot_ok, motion_fit, motion_frame,
                           metrics, passed, accept_demo, instruction, write_priors, build_tasks, compose_task, rescore)
    load.py                the agent tools' task, from $HDG_TASK (the task's config as JSON, set by the batch driver)
    push/                  task.py (PushTask), stage.py (kinematic staging), placement.py (object in a scene),
                           spots.py (push corridor), reward.py (metrics, hand-contact rule), prompts.py,
                           test_objects.py (GRScenes test set), hand_rescore.py
  datagen_pipelines/
    shared/                base.py (Stage); Stage classes every pipeline uses: priors.Prompts,
                           kimodo.MotionCandidates (all Kimodo use; device cpu | cuda), motions.SelectMotions,
                           leadin.LeadIns, datagen.PhysicsDemos; and compose.py, run.py (SONIC rollouts),
                           sampler.py (room geometry), library.py (asset library)
    agents/                wan.Videos, batch.AgentScenes (headless sessions), sandbox.py, cli.py + scene.py (the
                           agent's tools), push/SKILL.md (the push-scene skill)
    halo_infinigen/        push_clip.HaloClip, episodes.HaloEpisodes, rooms.InfinigenRooms, box_texture.py
  evaluators/              grscenes.GRScenesEval (build / run / rescore / viz; planners: HoldPlanner, vla.halo.HaloPlanner),
                           viz.py (review videos)
  vla/                     halo.py (HaloPlanner), export.py (HaloExport: demos -> HALO dataset), train.py (HaloTrain)
  scripts/                 datagen.py, train.py, eval.py
  utils/                   config (load_config, build), paths, sonic (GEAR-SONIC tracker), oak_cam, cost, stats,
                           scenes/ (spec, composer, geometry: composing the GRScenes homes)
```

Shared code never imports a task package: it calls the `Task` it was given, so a new task is a new `tasks/<name>/`
(a `Task` subclass) plus a `configs/task/<name>.yaml` and its run configs. Packages are namespace packages (no
`__init__.py`).

## Installation
1. Build the environment with `uv sync` (creates `.venv/`). Scene generation also needs
   [Claude Code](https://docs.claude.com/en/docs/claude-code) (`claude` on the PATH, `ANTHROPIC_API_KEY` set) and
   bubblewrap (`bwrap`, which needs unprivileged user namespaces).
2. Third-party repos go under `$THIRD_PARTY_ROOT` (default `utilities/repos` in this repo); organize them however
   you like and point the variable there. Each has its own environment; build it with the repo's own instructions.
    - Kimodo: `git clone https://github.com/nv-tlabs/kimodo $THIRD_PARTY_ROOT/kimodo` (commit `e1c219c`)
    - Wan 2.2: `git clone https://github.com/Wan-Video/Wan2.2 $THIRD_PARTY_ROOT/Wan2.2` (commit `1ea34ff`)
    - HALO (baseline pipeline, VLA training and eval): `git clone https://github.com/apple-aiml-research/ml-halo.git
      $THIRD_PARTY_ROOT/ml-halo` (commit `35d8334`); `train_halo.md` covers the generator environment.
    - Infinigen (HALO-Infinigen scenes only): `$THIRD_PARTY_ROOT/infinigen`.
3. Download checkpoints to `utilities/huggingface/hub`:
    ```
    export HF_HOME=$PWD/utilities/huggingface
    uv run hf download Wan-AI/Wan2.2-T2V-A14B                                    # scene videos
    uv run hf download nvidia/Kimodo-G1-RP-v1                                    # G1 push motions
    uv run hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp          # Kimodo text encoder
    uv run hf download McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised
    uv run hf download meta-llama/Meta-Llama-3-8B-Instruct                       # gated: needs an HF token with access
    uv run hf download laion/CLIP-ViT-L-14-laion2B-s32B-b82K                     # asset library search
    ```
4. Other downloads:
    - GEAR-SONIC motion tracker: `uv run hf download nvidia/GEAR-SONIC --local-dir utilities/checkpoints/gear_sonic`
      (or set `GEAR_SONIC_DIR`). The tracker reads `model_encoder.onnx`, `model_decoder.onnx` and
      `observation_config.yaml`.
    - Asset library objects (AI2-THOR + Objaverse home assets) from [MolmoSpaces](https://github.com/allenai/molmospaces):
      objects `thor` version `20251117` and `objaverse` version `20260131`, installed with
      `MLSPACES_CACHE_DIR=utilities/molmo-spaces-resources` (see its `docs/assets.md`). Only the 7,580 assets in the
      library index are used.
    - Asset library index (CLIP features, categories, sizes; 95 MB):
      `uv run hf download May00Mei/humanoid-datagen-library index.pkl --repo-type dataset --local-dir utilities/library`

Every shell that runs the pipeline needs:
```
export HF_HOME=$PWD/utilities/huggingface
export MUJOCO_GL=egl
export THIRD_PARTY_ROOT=$PWD/utilities/repos     # or wherever the repos are
```

## Pipeline

One config per run; a stage's config inherits its pipeline's `base.yaml` (the task and the shared stage settings:
prior, roots, sharding). Array jobs:
the prior stages take priors chunk `SLURM_ARRAY_TASK_ID` (`per_task` priors each); the scene stages take shard
`SLURM_ARRAY_TASK_ID` of `SLURM_ARRAY_TASK_COUNT` (or `shard` / `num_shards` in the config).

| Step | Config (`datagen.py` unless noted) | Env |
|---|---|---|
| 1. Prior prompts | `datagen/push/agents/prompts` | base |
| 2. Videos | `datagen/push/agents/videos` | Wan2.2, GPU |
| 3. Motion candidates | `datagen/push/agents/motions` | kimodo |
| 4. Motion selection | `datagen/push/agents/select` | base |
| 5. Agent scenes | `datagen/push/agents/scenes` (text prior: `agents_text/scenes`) | base + claude + bwrap |
| 6. Lead-ins | `datagen/push/agents/leadin` | kimodo, GPU |
| 7. Physics demos | `datagen/push/agents/demos` (text prior: `agents_text/demos`) | base |
| 8. Export | `train.py`: `train/push/export` | base |
| 9. Train | `train.py`: `train/push/halo_ours` (`halo_text`) | HALO generator |
| 10. Eval | `eval.py`: `eval/push/grscenes_halo_ours` (`_text`, `_halo_infinigen`, `grscenes_hold`) | base + HALO generator packages |
| 10b. Hand rescoring | `eval.py`: `eval/push/rescore` | base |
| (once) Test set | `eval.py`: `eval/push/build_grscenes` | base |

HALO-Infinigen baseline: `datagen/push/halo_infinigen/clip` (kimodo env) and `clip_check`, then `episodes` (kimodo env,
GPU) and `episodes_render`; `train/push/export_halo_infinigen`, `train/push/halo_halo_infinigen`;
`datagen/push/halo_infinigen/rooms` converts its rooms for our physics demos.

Kimodo stages run in the kimodo env from `$THIRD_PARTY_ROOT/kimodo` (so `kimodo` is the package there) and read the
`kimodo:` block or `device` of their stage: `cuda`, or `cpu` on a login node (the 15 GB text encoder then runs in its
own process before Kimodo loads; keep it to a few motions per call). Every env (base, kimodo, Wan2.2) needs `hydra-core` and
`omegaconf` for the entry scripts.

### Prior prompts and videos
`configs/datagen/push/prompts_1000.jsonl` holds 1,000 hand-written Wan prompts, each a different home and object
instance over 18 object types, half third-person and half egocentric. Step 1 writes
`outputs/push_box/priors_1000/p000..p999/` with `wan.txt`, `kimodo.txt` (Kimodo phrasing by object height) and
`meta.json`; every block of 18 priors covers all object types, so the first N priors make a balanced pilot. Step 2
loads Wan once per call, then takes ~25 min per video on one 80 GB GPU (`offload: true` on smaller GPUs).

### Motions
Step 3 samples 24 Kimodo candidates per prior (`<prior>/motion_candidates/`); step 4 keeps the first 4 whose palms
meet the object at a workable height, that GEAR-SONIC tracks without falling, and that push at least 1 m
(`<prior>/motions/m0..m3.csv`, `motion_select.json`, review videos in `motion_videos/`).

### Agent scenes
One fresh headless Claude Code session per prior (`claude -p "/push-scene ..."`, skill in
`humanoid_datagen/datagen_pipelines/agents/push/`). Each session runs in a bubblewrap sandbox that shows it only its
run dir, the tool code, the Python env, the tracker and the asset library (minus the held-out test assets in
`configs/eval/push/test_split.json`). It builds `room.xml` and `task.json`, checks them in physics with
GEAR-SONIC and finalizes `result.json`. Each run is capped at $30 and 250 turns (finished runs averaged about $13).
With a video prior the agent gets the Wan video, `wan.txt`, `kimodo.txt` and the motions; with a text prior, the same
agent, tools and budget, with `wan.txt` as the brief and no video. Reruns skip finalized runs; `dry_run: true` in the stage block
prints the session commands without launching anything.

### Demos, training and eval
Lead-ins (step 6) add a Kimodo walk-up from SONIC's standing pose into every selected motion. Step 7 places lead-in
plus motion at several spots per agent scene with jittered start, yaw, mass and friction, runs GEAR-SONIC physics,
and keeps demos that push the object at least 0.3 m without tipping it or falling, with the hands (not the legs)
doing the pushing (`hand_rule`). Accepted demos get a calibrated OAK-1 fisheye video. Steps 8–10 export matched
datasets per arm, train the HALO generator, and run it closed-loop with GEAR-SONIC in held-out GRScenes homes.
