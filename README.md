# KODIAK

[![CI](https://github.com/marjanstoimchev/KODIAK/actions/workflows/ci.yml/badge.svg)](https://github.com/marjanstoimchev/KODIAK/actions/workflows/ci.yml) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Paper](https://img.shields.io/badge/bioRxiv-10.64898%2F2026.10.03.756431-b31b1b.svg)](https://www.biorxiv.org/content/10.64898/2026.10.03.756431v1)

Official PyTorch Lightning implementation of **KODIAK**, the self-supervised method from

> **Discovering Latent Scientific Concepts through Discrete Representation Learning**
> Marjan Stoimchev, Monu Verma, Aveline Filliol, Maria Skamagki, Sara Flowers, Jacob Cohen, Zachary Azadian,
> Nancy Newlin, Mohamed Saeed Abdel-Mottaleb, Saso Dzeroski, Paul B. Romesser, Scott Lowe, Nevenka Dimitrova.
> *Discovery Science (DS 2026)*, Lecture Notes in Computer Science, Springer.
> [bioRxiv preprint](https://www.biorxiv.org/content/10.64898/2026.10.03.756431v1)

**KODIAK** (**KO**debook **DI**stillation for **A**daptation) adapts a vision foundation model (DINOv3 ViT-S/16)
to a new imaging domain with only a few thousand **unlabelled** images. Continuing a standard self-supervised
objective (DINO-style distillation of continuous features) on such small datasets often makes the model *worse*;
KODIAK replaces the continuous targets with **discrete, balanced codebook assignments**, which keeps adaptation
stable. It also works when training from scratch. The adapted encoder is then used for classification with few
labels, k-NN retrieval, or feature extraction.

## How it works (in one minute)

Two copies of the same ViT encoder are trained: a **student** (updated by gradient descent) and a **teacher**
(an exponential moving average of the student). Every training image is augmented into 2 large *global* crops and
8 small *local* crops. A learnable **codebook** of `K` prototype vectors (default `K = 128`) defines the discrete
target space; the teacher assigns every patch to codebook entries, and **Sinkhorn-Knopp** balancing makes sure all
entries stay in use.

| Loss | What it teaches |
|------|-----------------|
| **Mask loss** (core) | Patches of a global crop are hidden from the student (MAE-style sparse encoder + small decoder). For each hidden patch the student must predict the teacher's balanced codebook assignment, i.e. *which visual concept* was there. |
| **Cross-view CLS loss** | The image-level CLS token of every crop (global and local) is mapped by a small MLP head to a `K`-way distribution and must match the teacher's distribution for the global crops, so local details and the whole image agree. |
| **KoLeo** | Spreads CLS features apart in representation space so they do not collapse. |

Total loss: `L = L_mask + λ_CLS · L_CLS + λ_KoLeo · L_KoLeo` with `λ_CLS = 1.0`, `λ_KoLeo = 0.1`.

**Which losses to use.** The paper's ablations give a simple recipe: use the full objective when pretraining
**from scratch** (dropping the CLS or KoLeo loss costs up to 24 points on DTD), but when **adapting the DINOv3
foundation model** the mask loss alone carries the gain and the auxiliary losses are optional
(`--cls_weight 0 --koleo_weight 0`). Performance is robust to the codebook size (`K` from 64 to 4096).

Everything is configured through one YAML file per dataset and a handful of command-line flags.

## Installation

Python 3.10+ and PyTorch 2.1+.

```bash
git clone https://github.com/marjanstoimchev/KODIAK.git
cd KODIAK
python -m venv .venv && source .venv/bin/activate   # or a conda env
pip install torch torchvision                        # choose the build for your CUDA version at https://pytorch.org
pip install -r requirements.txt
```

Check that everything works (one batch, runs on CPU too):

```bash
python scripts/train.py --config configs/eurosat/pretrain.yaml --fast_dev_run --num_workers 0 --precision 32
```

Notes:
- Run every command from the repository root; nothing needs to be installed as a package.
- `requirements-lock.txt` is the exact environment used for the paper (also used by `Singularity.def`).
- Weights & Biases logging is optional: `pip install wandb`.
- The DINOv3 ViT-S/16 weights used for *continued* pretraining ship in `dinov3_weights/` under the
  [DINOv3 license](dinov3/LICENSE.md).

## Quick start

The full workflow is **pretrain → fine-tune (or linear probe) → evaluate**. The example below uses EuroSAT
(10 classes, downloaded automatically from HuggingFace) on one GPU.

**1. Pretrain**, starting from the DINOv3 weights ("continued" pretraining, 100 epochs):

```bash
python scripts/train.py --config configs/eurosat/pretrain_continued.yaml \
    --devices 0 --max_epochs 100 --batch_size 128 --num_prototypes 128
```

The checkpoint lands in `checkpoints/pretraining/eurosat/<experiment>/last.ckpt`. To pretrain from random weights
instead, use `configs/eurosat/pretrain.yaml` (the paper uses 500 epochs).

**2. Train a classifier** on top of the pretrained encoder. Fine-tuning updates the whole network; add
`--freeze_backbone` for a linear probe:

```bash
python scripts/train_classifier.py --config configs/eurosat/classify.yaml \
    --pretrained_path checkpoints/pretraining/eurosat \
    --devices 0 --max_epochs 50 --learning_rate 1e-4 --seed 0
```

Test accuracy, F1, precision/recall and AUROC are written to `test_results.json` next to the logs.

**3. Evaluate the representation directly** (no training) with weighted k-NN:

```bash
python scripts/eval_knn.py --config configs/eurosat/classify.yaml \
    --pretrained_path checkpoints/pretraining/eurosat --seeds "0 1 42"
```

**4. Few-shot evaluation** (k labelled images per class):

```bash
python scripts/eval_lowshot.py --config configs/eurosat/classify.yaml \
    --pretrained_path checkpoints/pretraining/eurosat --k_shot 4 --label_seed 0 --train_seed 42
python scripts/aggregate_lowshot.py --input_dir output/lowshot      # mean ± std tables
```

Multiple GPUs: pass `--devices 0,1,2,3` (the batch size is the *total* batch size, split across GPUs).
The shell wrappers in `scripts/` (`run_sweep.sh`, `run_knn.sh`, `run_lowshot.sh`, `run_ablation.sh`) chain these
steps and loop over seeds; `scripts/slurm/` has the SLURM versions. All of them are documented with copy-paste
commands in [EXPERIMENTS.md](EXPERIMENTS.md).

### Using your own images

Put your images in one folder per class and point KODIAK at it. Both a flat layout and one sub-folder per slide
are detected automatically:

```
/data/myimages/
├── class_a/  img1.png img2.png ...
└── class_b/  ...
```

```bash
export KODIAK_DATA_DIR=/data                      # configs/pancreatic/*.yaml use ${KODIAK_DATA_DIR}/pancreatic
python scripts/train.py --config configs/pancreatic/pretrain_continued.yaml --root_dir /data/myimages
python scripts/train_classifier.py --config configs/pancreatic/classify.yaml --root_dir /data/myimages \
    --pretrained_path checkpoints/pretraining/pancreatic
```

Copy `configs/pancreatic/` to a new folder and set `data.num_classes` (classification) and the image size to
match your data. Details and the HuggingFace dataset wrapper are in [DATASETS.md](DATASETS.md).

## Datasets and configs

Each dataset has three configs: `pretrain.yaml` (from scratch), `pretrain_continued.yaml` (from DINOv3 weights)
and `classify.yaml` (fine-tuning, linear probing, k-NN, few-shot).

| Dataset | Config folder | Classes | Source |
|---------|---------------|---------|--------|
| Pancreatic (histopathology, local images) | `configs/pancreatic/` | 12 | local folder, `$KODIAK_DATA_DIR/pancreatic` |
| CIFAR-100 | `configs/cifar100/` | 100 | `uoft-cs/cifar100` |
| DTD | `configs/DTD/` | 47 | `cansa/Describable-Textures-Dataset-DTD` |
| EuroSAT | `configs/eurosat/` | 10 | `blanchon/EuroSAT_RGB` |
| Oxford Pets | `configs/oxford_pets/` | 37 | `timm/oxford-iiit-pet` |
| NCT-CRC-HE-100K | `configs/NCTCRCHE100K/` | 9 | `DykeF/NCTCRCHE100K` |
| ImageNet-1K | `configs/imagenet1k/` | 1000 | `ILSVRC/imagenet-1k` |

Recommended settings from the paper:

| Setting | Value |
|---------|-------|
| Pretraining from scratch | 500 epochs |
| Continued pretraining (from DINOv3 weights) | 100 epochs |
| Fine-tuning | 100 epochs (scratch) / 50 epochs (continued), LR 1e-4 |
| Linear probing | 50 epochs, LR 1e-3 |
| Batch size | 128 (HuggingFace datasets), 64 (Pancreatic, 256x256 images) |
| Prototypes | 128 (sweep: 64 ... 4096, see [ABLATIONS.md](ABLATIONS.md)) |

## Where results go

```
checkpoints/pretraining/{dataset}/{experiment}/last.ckpt          # pretrained encoder (+ periodic checkpoints with --save_every_n_epochs)
logs/pretraining/{dataset}/{experiment}/                          # config.yaml, metrics.csv, training_summary.json
logs/classification/{dataset}/{pretrain_info}/{mode}_seed_{seed}/ # test_results.json, training_summary.json
output/knn/{dataset}/{pretrain_info}/knn_results.json
output/lowshot/{dataset}/{method}/{mode}_{k}shot_ls{label_seed}_ts{train_seed}/test_results.json
```

`{experiment}` encodes the key hyper-parameters, e.g. `kodiak_eurosat_continued_proto128_koleo0.1_cls1.0_mc8`,
and `{pretrain_info}` is read back from the checkpoint, e.g. `proto128_koleo0.1_cls1_mc8`.

`test_results.json` contains `test_acc`, `test_acc_top5` (if more than 5 classes), `test_f1_macro`,
`test_f1_weighted`, `test_precision_macro`, `test_recall_macro` and `test_auroc`. The classifier keeps the
checkpoint with the lowest validation loss and evaluates that one on the test set.

Logging goes to CSV by default; `--logger tensorboard` or `--logger wandb` switch the backend
(`tensorboard --logdir logs`).

---

## Command-line reference

### `train.py` — Pretraining

Self-supervised pretraining with KODIAK.

```bash
python scripts/train.py \
    --config configs/pancreatic/pretrain.yaml \
    --devices 0,1 \
    --batch_size 64 \
    --max_epochs 500 \
    --num_prototypes 128 \
    --multi_crop \
    --logger tensorboard
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--devices` | GPU indices, comma-separated | from config |
| `--batch_size` | Total batch size | from config |
| `--learning_rate` | Peak learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--num_prototypes` | Number of prototype vectors | from config |
| `--koleo_weight` | KoLeo regularization weight | from config |
| `--cls_weight` | Multi-crop CLS distillation weight | from config |
| `--multi_crop` | Enable multi-crop (2 global + 8 local) | off |
| `--no_sinkhorn` | Use softmax instead of Sinkhorn-Knopp | off |
| `--freeze_cls_head` | Keep the CLS prototype head at its random init (reproduces runs made before the head was trainable) | off |
| `--compile` | Enable `torch.compile()` | off |
| `--precision` | Training precision | bf16-mixed |
| `--logger` | `csv`, `tensorboard`, or `wandb` | csv |
| `--name` | Experiment name | from config |
| `--resume` | Resume from a Lightning checkpoint (`last.ckpt`) | — |
| `--pretrained_path` | Initialize from pretrained weights | — |
| `--save_every_n_epochs` | Save a checkpoint every N epochs (enables `last.ckpt` updates for `--resume`) | from config (off) |
| `--seed` | Random seed | from config |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |
| `--fast_dev_run` | Quick sanity check (1 batch) | off |

Without `--save_every_n_epochs` only the final `last.ckpt` is written when training completes.

### `train_classifier.py` — Classification

Fine-tuning or linear evaluation using a pretrained KODIAK encoder.

```bash
python scripts/train_classifier.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --devices 0,1 \
    --max_epochs 100 \
    --learning_rate 1e-4 \
    --batch_size 64 \
    --seed 42 \
    --freeze_backbone \
    --logger tensorboard
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint file or directory | — |
| `--checkpoint_type` | `last` or `best` (lowest loss) | last |
| `--encoder_type` | `teacher` or `student` | teacher |
| `--freeze_backbone` | Freeze encoder for linear evaluation | off |
| `--use_cls_token` | Use CLS token or mean patch pooling | from config |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens (2x features) | off |
| `--devices` | GPU indices | from config |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | from config |
| `--precision` | Training precision | bf16-mixed |
| `--seed` | Random seed | from config |
| `--logger` | `csv`, `tensorboard`, or `wandb` | csv |
| `--eval_only` | Test evaluation only (no training) | off |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |
| `--fast_dev_run` | Quick sanity check | off |

After training, the checkpoint with the lowest `val_loss` is evaluated on the test set and the metrics are written to `test_results.json`.

### `eval_knn.py` — k-NN Evaluation

Evaluates feature quality using weighted k-NN classification (no training). Follows the DINO/DINOv3 protocol.

```bash
python scripts/eval_knn.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --k 20 \
    --temperature 0.07 \
    --batch_size 256 \
    --seeds "0 1 42"
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint **(required)** | — |
| `--checkpoint_type` | `last` or `best` | last |
| `--init_mode` | `scratch` or `continued` | scratch |
| `--encoder_type` | `teacher` or `student` | teacher |
| `--k` | Number of nearest neighbors | 20 |
| `--temperature` | Softmax temperature for voting | 0.07 |
| `--batch_size` | Feature extraction batch size | 256 |
| `--devices` | GPU device index (single GPU) | 0 |
| `--concat_cls_patch` | Concatenate CLS + mean patch tokens | off |
| `--output_base_dir` | Base output directory | output |
| `--seeds` | Evaluation seeds, space-separated | "42" |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |

### `eval_lowshot.py` — K-Shot Evaluation

Trains a classifier with limited labeled data (k examples per class).

```bash
python scripts/eval_lowshot.py \
    --config configs/pancreatic/classify.yaml \
    --pretrained_path output/checkpoints/pretraining/pancreatic/last.ckpt \
    --k_shot 4 \
    --label_seed 0 \
    --train_seed 42
```

| Flag | Description | Default |
|------|-------------|---------|
| `--config` | Path to classify YAML config **(required)** | — |
| `--pretrained_path` | Pretrained checkpoint **(required)** | — |
| `--k_shot` | Labeled examples per class **(required)** | — |
| `--label_seed` | Seed for label subset selection **(required)** | — |
| `--train_seed` | Seed for training **(required)** | — |
| `--freeze_backbone` | Freeze backbone (linear probing) | off |
| `--batch_size` | Batch size | from config |
| `--learning_rate` | Learning rate | from config |
| `--max_epochs` | Training epochs | 100 |
| `--output_dir` | Base output directory | output/lowshot |
| `--method_tag` | Sub-folder name for the method | auto from checkpoint |
| `--root_dir` | Folder-based dataset root (overrides `data.root_dir`) | from config |
| `--num_workers` | DataLoader workers (overrides `data.num_workers`) | from config |

The baseline methods handled by `scripts/run_lowshot.sh` (`dinov3`, `mae`, `ijepa`, `moca`) are not part of this
repository. Point `DINOV3_WEIGHTS`, `DINOV3_CONTINUED_DIR`, `MAE_OUTPUT_DIR`, `IJEPA_OUTPUT_DIR` and `MOCA_OUTPUT_DIR`
at your own checkpoints; methods whose variable is unset are skipped.

---

---

## Configuration

Configs are YAML files with sections that can be overridden via CLI arguments.

### Key Hyperparameters

```yaml
model:
  num_prototypes: 4096              # Number of prototype vectors
  projector_dim: 256                # Projection head dimension
  mask_ratio_min_max: [0.1, 0.5]   # Masking ratio range

optimizer:
  learning_rate: 1.0e-4             # Peak LR (with sqrt_wrt_1024 scaling)
  min_lr: 1.0e-6                    # Final LR after cosine decay
  warmup_epochs: 10                 # Linear warmup
  weight_decay: 0.04                # Start weight decay
  weight_decay_end: 0.4             # End weight decay (cosine)
  ema_momentum: 0.992               # Teacher EMA start
  ema_momentum_end: 1.0             # Teacher EMA end
  layerwise_decay: 0.9              # Layer-wise LR decay

loss:
  koleo_loss_weight: 0.1            # KoLeo regularization weight
  prototype_cls_loss_weight: 1.0    # Multi-crop CLS distillation weight
  teacher_temp: 0.04                # Teacher softmax temperature
  sinkhorn_iters: 3                 # Sinkhorn-Knopp iterations

training:
  precision: bf16-mixed
  gradient_clip_val: 3.0
  strategy: ddp
```

---

---

## Project structure

```
KODIAK/
├── src/
│   ├── models/        kodiak.py (teacher-student model), components.py (ViT encoder, decoder, heads), classifier.py
│   ├── losses/        MaskLoss + Sinkhorn-Knopp, KoLeo, MultiCropPrototypeCLSLoss
│   ├── learners/      PyTorch Lightning modules for pretraining and classification
│   ├── data/          datamodules, DINOv3 augmentations, masking, HuggingFace + image-folder datasets
│   ├── utils/         YAML config loading (env-var expansion, CLI overrides), validation, loggers, device helpers
│   └── callbacks/     training timer, progress bar
├── scripts/           train.py, train_classifier.py, eval_knn.py, eval_lowshot.py, eval.py, aggregate_lowshot.py
│   ├── run_*.sh       multi-step / multi-seed wrappers
│   └── slurm/         SLURM job scripts (optional Singularity container)
├── configs/           one folder per dataset
├── tests/             pytest suite (unit + end-to-end CLI tests, CPU only)
├── dinov3/            vendored DINOv3 library (backbone layers)
├── dinov3_weights/    DINOv3 ViT-S/16 weights for continued pretraining
├── EXPERIMENTS.md     every experiment as copy-paste commands (local + SLURM)
├── DATASETS.md        dataset details and per-dataset commands
└── ABLATIONS.md       ablation studies and prototype-count sweep
```

## Development

```bash
pip install -r requirements-dev.txt
make lint        # ruff
make test-fast   # unit tests (seconds)
make test        # unit + end-to-end CLI tests on CPU (about a minute)
```

The end-to-end tests run every entry point on a tiny synthetic image folder with a 1-block ViT, so no GPU or
dataset download is needed. The same suite runs in GitHub Actions on every push and pull request.

## License

The KODIAK code (`src/`, `scripts/`, `configs/`, `tests/`) is released under the [MIT License](LICENSE).
The vendored DINOv3 library (`dinov3/`) and the pretrained weights in `dinov3_weights/` are distributed separately
under the [DINOv3 License Agreement](dinov3/LICENSE.md).

## Citation

If you use KODIAK, please cite:

```bibtex
@inproceedings{Stoimchev26,
  author    = {Stoimchev, Marjan and Verma, Monu and Filliol, Aveline and Skamagki, Maria and Flowers, Sara and Cohen, Jacob and Azadian, Zachary and Newlin, Nancy and Abdel-Mottaleb, Mohamed Saeed and Dzeroski, Saso and Romesser, Paul B. and Lowe, Scott and Dimitrova, Nevenka},
  title     = {Discovering Latent Scientific Concepts through Discrete Representation Learning},
  booktitle = {Discovery Science (DS 2026)},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  address   = {Mainz, Germany},
  year      = {2026},
  month     = oct
}
```

Preprint: [bioRxiv 10.64898/2026.10.03.756431](https://www.biorxiv.org/content/10.64898/2026.10.03.756431v1).
