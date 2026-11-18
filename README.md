# Humanoid Data Generation

Push-task scenes for the Unitree G1: a Wan2.2 video of a person pushing a household object in a home, Kimodo G1
push motions, and a Claude agent that rebuilds the room as a MuJoCo scene and stages the robot's push in it.

**Run every command from the repo root.** Config paths are relative to it, and the configs live in
`humanoid_datagen/configs/` (`--config_name push_box/<name>` reads `humanoid_datagen/configs/push_box/<name>.yaml`).

## Installation
1. Build environment with `uv sync` (creates `.venv/`).
   Scene generation also needs [Claude Code](https://docs.claude.com/en/docs/claude-code) (`claude` on the PATH,
   `ANTHROPIC_API_KEY` set) and bubblewrap (`bwrap`, which needs unprivileged user namespaces).
2. Clone third-party repos to `utilities/repos`. Each has its own environment; build it with the repo's own
   instructions.
    - Kimodo: `git clone https://github.com/nv-tlabs/kimodo utilities/repos/kimodo` (we use commit `e1c219c`)
    - Wan 2.2: `git clone https://github.com/Wan-Video/Wan2.2 utilities/repos/Wan2.2` (we use commit `1ea34ff`)
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
    - GEAR-SONIC motion tracker: `uv run hf download nvidia/GEAR-SONIC --local-dir utilities/checkpoints/gear_sonic`.
      The tracker reads `model_encoder.onnx`, `model_decoder.onnx` and `observation_config.yaml` from that folder.
    - Asset library objects (AI2-THOR + Objaverse home assets) from [MolmoSpaces](https://github.com/allenai/molmospaces):
      objects `thor` version `20251117` and `objaverse` version `20260131`, installed with
      `MLSPACES_CACHE_DIR=utilities/molmo-spaces-resources` (see its `docs/assets.md`), so that they sit in
      `utilities/molmo-spaces-resources/objects/{thor/20251117,objaverse/20260131}/`. Only the 7,580 assets in the
      library index are used.
    - Asset library index (CLIP features, categories, sizes; 95 MB):
      `uv run hf download May00Mei/humanoid-datagen-library index.pkl --repo-type dataset --local-dir utilities/library`

Every shell that runs the pipeline needs:
```
export HF_HOME=$PWD/utilities/huggingface
export MUJOCO_GL=egl
```

Layout after installation:
```
utilities/
  repos/kimodo/  repos/Wan2.2/
  huggingface/hub/            HF model cache
  checkpoints/gear_sonic/     GEAR-SONIC ONNX tracker
  molmo-spaces-resources/     asset library objects
  library/index.pkl           asset library index
```

## Video Generation
1. Write the 1000 prior folders from the prompt list (`humanoid_datagen/configs/push_box/prompts_1000.jsonl`: one
   Wan prompt per prior, each a different home and object instance over 18 object types, half third-person, half
   egocentric):
    ```
    uv run python -m humanoid_datagen.push_box.prompts --config_name push_box/prompts
    ```
    Output: `outputs/push_box/priors_1000/p000..p999/` with `wan.txt`, `kimodo.txt` and `meta.json`. Every block of
    18 priors covers all object types, so the first N priors make a balanced pilot.
2. One Wan2.2 video per prior, with the Wan environment. The model loads once per call, then ~25 min per video on one
   80 GB GPU (add `--offload` on smaller GPUs). Split the priors across GPUs by range, e.g. `p00[0-3]`:
    ```
    jobs=(); for d in outputs/push_box/priors_1000/p00[0-3]; do
      cam=$(uv run python -c "import json,sys; print(json.load(open(sys.argv[1]))['camera'])" $d/meta.json)
      jobs+=(--job $d/wan.txt $d/wan_$cam.mp4)
    done
    <wan env python> wan_gen/run_t2v.py "${jobs[@]}" --seeds 0 --skip-existing
    ```
    Output: `<prior>/wan_<third|ego>_seed0.mp4`. `--skip-existing` makes reruns resume.

## Motion Generation
1. Kimodo G1 push motions, 24 candidates per prior, with the Kimodo environment on a GPU (the LLM2Vec text encoder
   needs ~15 GB):
    ```
    LOCAL_CACHE=True <kimodo env python> humanoid_datagen/push_box/kimodo_cpu.py samples tmp/kimodo_emb_0 \
        outputs/push_box/priors_1000/p00[0-3] --n 24 --subdir motion_candidates --prefix c --device cuda --seed 0
    ```
    Use a different embedding scratch dir (`tmp/kimodo_emb_<i>`) for each parallel run. Output:
    `<prior>/motion_candidates/c00..c23.csv`; priors that already have 24 are skipped.
2. Keep the first 4 candidates whose palms meet the object at a workable height, that GEAR-SONIC tracks without
   falling, and that push at least 1 m:
    ```
    uv run python -m humanoid_datagen.push_box.select_motions --config_name push_box/motions_1000
    ```
    Output: `<prior>/motions/m0..m3.csv`, `motion_select.json` and review videos `motion_videos/`. This runs over
    every prior; to split it, set `SLURM_ARRAY_TASK_ID=<i>` to process only priors `4i..4i+3`. A prior with no
    usable motion is skipped by scene generation.

## Scene Generation
One fresh headless Claude Code session per prior (`claude -p "/push-scene ..."`, skill in
`.claude/skills/push-scene/`). Each session runs inside a bubblewrap sandbox that shows it only its run dir, the
tool code, the Python env, the tracker and the asset library (minus the held-out test assets in
`humanoid_datagen/configs/push_box/test_split.json`). It builds `room.xml` and `task.json`, checks them in physics
with GEAR-SONIC, and finalizes `result.json`. Each run is capped at $30 and 250 turns; finished runs so far averaged
about $13.

- With Video: the agent gets the Wan video, `wan.txt` and `kimodo.txt`, and the motions.
    ```
    uv run python -m humanoid_datagen.scripts.run_build_scene --config_name push_box/agent_batch
    ```
    Output: `outputs/push_box/scenes/<prior>_s0/`.
- Without Video: the same agent, tools and budget, with `wan.txt` as a text brief and no video.
    ```
    uv run python -m humanoid_datagen.scripts.run_build_scene --config_name push_box/agent_batch_text
    ```
    Output: `outputs/push_box/scenes_text/<prior>_s0/`.

Both read the priors in `outputs/push_box/priors_1000/` that have a video (with-video arm) and selected motions.
Reruns skip runs with a finalized `result.json` and retry the others. Append `dry_run=true` to print the session
commands without launching anything. To split the priors across machines, give each machine its own copy of the
config with a disjoint `jobs.push.prior_ids` list, and set `max_concurrent` (sessions at once) to what the machine
can run; every session also runs MuJoCo physics and renders on the CPU.
