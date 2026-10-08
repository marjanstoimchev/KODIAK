# Quick start

### Installation

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
  [DINOv3 license](../dinov3/LICENSE.md).

### Quick start

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

#### Using your own images

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

### Datasets and configs

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

### Where results go

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

## Next steps

- Every experiment from the paper as copy-paste commands (local and SLURM): [EXPERIMENTS.md](EXPERIMENTS.md)
- Per-dataset details and the HuggingFace wrapper: [DATASETS.md](DATASETS.md)
- All command-line flags and config keys: [CLI.md](CLI.md)
- Ablations and the prototype-count sweep: [ABLATIONS.md](ABLATIONS.md)
