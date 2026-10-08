# Development

```bash
pip install -r requirements-dev.txt
make lint        # ruff
make test-fast   # unit tests (seconds)
make test        # unit + end-to-end CLI tests on CPU (about a minute)
```

The end-to-end tests run every entry point on a tiny synthetic image folder with a 1-block ViT, so no GPU or
dataset download is needed. The same suite runs in GitHub Actions on every push and pull request.

### Project structure

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

## Adding a dataset

1. HuggingFace: copy a config folder (e.g. `configs/eurosat/`), change `data.hf_dataset_name`, `data.name`,
   `experiment.name` and `data.num_classes`.
2. Image folders: copy `configs/pancreatic/`, set `data.root_dir` (or use `--root_dir`) and `data.num_classes`.
3. Anything else: implement a `torch.utils.data.Dataset` returning `{"image": PIL.Image, "label": int}` and register it
   with `@DatasetRegistry.register("my_dataset")` (see [`src/data/utils/factory.py`](../src/data/utils/factory.py)),
   then use `dataset_type: my_dataset` in the config.
