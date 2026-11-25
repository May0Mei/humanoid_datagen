# Training the HALO generator on the HALO push dataset

This trains HALO's vision-language motion generator (`ml-halo/generator`) on our HALO-baseline push dataset. The
dataset was made with HALO's own data pipeline:

| | |
|---|---|
| Scenes | 50 Infinigen scenes from HALO's `render_scenes_1000.txt` list |
| Episodes | 50 per scene: 2,500 generated, of which 2,376 pass HALO's `motion_check` gate. Only those are in the dataset. |
| Motion | The G1 starts standing 1.3 m behind the box (pelvis to box centre). One Kimodo call walks it up and bends it into a fixed Kimodo push clip, which stands in for HALO's unreleased motion-capture push. The robot pushes the box about 2 m. Each episode is 221 frames (7.4 s) at 30 fps. |
| Video | OAK-1 head camera on the helmet, rendered with HALO's MuJoCo backend and warped to fisheye with our calibrated intrinsics (1280×800). Clean renders: no appearance randomization (`cam_fx`). |
| Instruction | "Push the cardboard box forward." |
| Size | 3.9 GB: 525,096 frames |

## 1. Clone HALO and set up the generator environment

```bash
git clone https://github.com/apple-aiml-research/ml-halo.git
cd ml-halo
git checkout 35d8334            # the commit we trained with
cd generator
bash setup.sh                   # conda env halo_generator (Python 3.10, CUDA 12.8 wheels)
conda activate halo_generator
```

The backbone, `Qwen/Qwen3.5-2B`, downloads from Hugging Face on the first run into `$HF_HOME`. Run every command
below from `ml-halo/generator/`.

## 2. Download the dataset

```bash
hf download <HF_DATASET_REPO> --repo-type dataset --local-dir /path/to/halo_push_infinigen
```

The download is a dataset in HALO's layout, which the generator reads directly:
* `data/episode_NNNNNN.parquet`: pelvis position and orientation, and the 29 joint positions per frame;
* `videos/observation.images.fpv_cam/episode_NNNNNN.mp4`: the fisheye head-camera video;
* `meta/`: `info.json`, `episodes.jsonl` and `tasks.jsonl`. `sources.jsonl` maps each episode to our episode id.

On its first run the trainer decodes every video to 224×224 frames in `/path/to/halo_push_infinigen/frames/`, which
takes about 80 GB of disk. Later runs on the same folder reuse them.

## 3. Config

Save this as `configs/push_halo_infinigen.yaml`. It is the config of our run. Set `data_dir`, `checkpoint_dir`, and
`batch_size` (see below).

```yaml
experiment_name: "halo_push_halo_infinigen"
seed: 42
device: "cuda"

data:
  data_dir: "/path/to/halo_push_infinigen"
  image_size: [224, 224]
  trajectory_length: 150
  train_split: 0.98
  batch_size: 32              # PER GPU: 256 / number of GPUs (8 GPUs -> 32; we used 4 x 64)
  num_workers: 0
  pin_memory: true
  normalize_motion: true
  cache_frames: true          # every GPU process holds all frames in RAM: ~79 GB each (see below)
  streaming_chunk_size: 0     # local dataset
  cold_start_augment: true

model:
  vlm_backbone: "qwen3.5-2b"
  vlm_frozen: true
  vlm_dtype: "bfloat16"
  vlm_vision_trainable: false
  cond_pos_embed: true
  max_cond_tokens: 96
  vlm_gpu_preprocess: true
  num_inference_steps: 15
  hidden_dim: 512
  num_decoder_layers: 8
  num_heads: 8
  dropout: 0.1
  trajectory_length: 150
  output_dim: 36
  inpaint_prev_chunk: true
  prev_chunk_frames: 15

training:
  num_epochs: 50
  accumulation_steps: 1
  learning_rate: 0.0003
  vlm_lr: 0.00001
  weight_decay: 0.01
  betas: [0.9, 0.999]
  max_grad_norm: 1.0
  checkpoint_dir: "/path/to/checkpoints/halo_push_halo_infinigen"
  checkpoint_every: 5
  eval_every: 0
```

**Batch size.** We train every push data source with a global batch of 256, so that runs are comparable.
`batch_size` is per GPU: set it to 256 divided by the number of GPUs. HALO's own configs use a global batch of 5,120
(64 GPUs × 80). Don't use that here; it changes the optimization.

**Memory.** With `cache_frames: true` each GPU process keeps its own copy of all frames, about 79 GB of host RAM per
GPU: 8 GPUs need about 630 GB. If the node has less, set `cache_frames: false`. Frames are then read from
`frames/*.pt` on disk as needed, which is slower.

## 4. Train

One node, 8 GPUs (with `batch_size: 32`):

```bash
torchrun --standalone --nproc_per_node 8 scripts/train.py --config configs/push_halo_infinigen.yaml --wandb --wandb_project <project>
```

Drop `--wandb ...` to log only to `train.log` and TensorBoard in `checkpoint_dir`. On a single GPU, use
`batch_size: 256` and run `python scripts/train.py --config configs/push_halo_infinigen.yaml`.

**What to expect** (our run: 4 × H200, 64 per GPU):
* 549,521 training samples, 2,147 steps per epoch;
* about 15 min per epoch (2.4 steps/s), so about 13 h for 50 epochs;
* about 10 min of start-up (frame decoding and loading);
* training loss about 0.12–0.17 early in epoch 0.

Outputs go to `checkpoint_dir`: `checkpoint_epoch_N.pt` every 5 epochs, `best_model.pt`, `final_model.pt`,
`config.json` and `train.log`.
