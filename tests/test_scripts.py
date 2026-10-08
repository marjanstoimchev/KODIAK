"""End-to-end tests for the CLI entry points on CPU with a tiny synthetic dataset.

The pipeline exercised here is the one documented in the README:
pretrain -> resume -> classify (fine-tune + linear eval) -> k-NN -> low-shot -> aggregate.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"

pytestmark = pytest.mark.integration


def run(args, cwd=ROOT, timeout=600):
    env = dict(os.environ, PYTHONUNBUFFERED="1", CUDA_VISIBLE_DEVICES="")
    proc = subprocess.run([sys.executable, *map(str, args)], cwd=cwd, env=env, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        print(proc.stdout[-6000:])
        print(proc.stderr[-6000:])
    assert proc.returncode == 0, f"command failed: {' '.join(map(str, args))}"
    return proc.stdout + proc.stderr


@pytest.fixture(scope="module")
def pretrain_run(tiny_pretrain_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("pretrain")
    ckpt_base, log_base = out / "ckpt", out / "logs"
    log = run([SCRIPTS / "train.py", "--config", tiny_pretrain_config, "--max_epochs", "2",
               "--save_every_n_epochs", "1", "--checkpoint_base_dir", ckpt_base, "--log_base_dir", log_base,
               "--num_prototypes", "16", "--koleo_weight", "0.1", "--cls_weight", "1.0"])
    exp_dir = ckpt_base / "pancreatic" / "tiny_proto16_koleo0.1_cls1.0_mc2"
    return {"log": log, "ckpt_dir": exp_dir, "ckpt": exp_dir / "last.ckpt", "log_base": log_base, "out": out}


class TestPretrain:
    def test_checkpoint_and_logs_written(self, pretrain_run):
        ckpt_dir = pretrain_run["ckpt_dir"]
        assert pretrain_run["ckpt"].exists()
        periodic = [p for p in ckpt_dir.glob("*.ckpt") if p.name != "last.ckpt"]
        assert len(periodic) >= 1, "periodic checkpoints requested via --save_every_n_epochs"
        state = torch.load(pretrain_run["ckpt"], map_location="cpu", weights_only=False)
        assert state["hyper_parameters"]["num_prototypes"] == 16
        assert state["hyper_parameters"]["n_local_crops"] == 2
        assert state["global_step"] == 12  # 24 images / batch 4 = 6 steps x 2 epochs
        log_dir = pretrain_run["log_base"] / "pancreatic" / "tiny_proto16_koleo0.1_cls1.0_mc2"
        assert (log_dir / "config.yaml").exists()
        assert (log_dir / "training_summary.json").exists()
        assert "Configuration validated successfully" in pretrain_run["log"]

    def test_resume(self, pretrain_run, tiny_pretrain_config, tmp_path):
        run([SCRIPTS / "train.py", "--config", tiny_pretrain_config, "--max_epochs", "3",
             "--resume", pretrain_run["ckpt"], "--checkpoint_base_dir", tmp_path / "ckpt",
             "--log_base_dir", tmp_path / "logs", "--num_prototypes", "16"])
        resumed = tmp_path / "ckpt" / "pancreatic" / "tiny_proto16_koleo0.1_cls1.0_mc2" / "last.ckpt"
        state = torch.load(resumed, map_location="cpu", weights_only=False)
        assert state["global_step"] == 18  # 12 restored + one more epoch of 6 steps
        assert state["epoch"] == 3

    def test_fast_dev_run_and_legacy_mode(self, tiny_pretrain_config, tmp_path):
        log = run([SCRIPTS / "train.py", "--config", tiny_pretrain_config, "--fast_dev_run", "--no_multi_crop",
                   "--no_sinkhorn", "--cls_weight", "0", "--checkpoint_base_dir", tmp_path / "c",
                   "--log_base_dir", tmp_path / "l"])
        assert "Multi-Crop:         False" in log

    def test_invalid_config_fails_cleanly(self, tiny_pretrain_config, tmp_path):
        proc = subprocess.run([sys.executable, str(SCRIPTS / "train.py"), "--config", str(tiny_pretrain_config),
                               "--root_dir", str(tmp_path / "does_not_exist"), "--fast_dev_run"],
                              cwd=ROOT, capture_output=True, text=True, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""))
        assert proc.returncode == 1
        assert "CONFIGURATION ERRORS" in proc.stdout


@pytest.fixture(scope="module")
def classify_runs(pretrain_run, tiny_classify_config, tmp_path_factory):
    out = tmp_path_factory.mktemp("classify")
    runs = {}
    for mode, extra in [("finetune", []), ("lineareval", ["--freeze_backbone"])]:
        log_dir, ckpt_dir = out / mode / "logs", out / mode / "ckpt"
        log = run([SCRIPTS / "train_classifier.py", "--config", tiny_classify_config,
                   "--pretrained_path", pretrain_run["ckpt_dir"], "--max_epochs", "2", "--seed", "1",
                   "--log_dir", log_dir, "--checkpoint_dir", ckpt_dir, *extra])
        runs[mode] = {"log": log, "log_dir": log_dir, "ckpt_dir": ckpt_dir}
    return runs


class TestClassify:
    @pytest.mark.parametrize("mode", ["finetune", "lineareval"])
    def test_results_json(self, classify_runs, mode):
        r = classify_runs[mode]
        results = json.loads((r["log_dir"] / "test_results.json").read_text())
        assert results["mode"] == mode and results["seed"] == 1
        assert results["pretrain_num_prototypes"] == 16
        for key in ["test_acc", "test_f1_macro", "test_auroc", "test_loss"]:
            assert isinstance(results[key], float)
        assert results["test_acc_top5"] is None  # only 3 classes
        assert (r["ckpt_dir"] / "last.ckpt").exists()
        assert any("val_loss" in p.name for p in r["ckpt_dir"].glob("classifier-*.ckpt"))
        assert "Testing best checkpoint" in r["log"]
        assert "Loaded KODIAK teacher_encoder weights" in r["log"]
        assert "Backbone frozen" in r["log"] if mode == "lineareval" else "Backbone frozen" not in r["log"]

    def test_eval_only(self, classify_runs, tiny_classify_config):
        r = classify_runs["finetune"]
        log = run([SCRIPTS / "train_classifier.py", "--config", tiny_classify_config, "--eval_only",
                   "--log_dir", r["log_dir"], "--checkpoint_dir", r["ckpt_dir"]])
        assert "EVAL-ONLY MODE" in log and "Test Evaluation Complete" in log

    def test_standalone_eval_script(self, classify_runs, tiny_classify_config, tmp_path):
        r = classify_runs["finetune"]
        run([SCRIPTS / "eval.py", "--config", tiny_classify_config, "--checkpoint", r["ckpt_dir"],
             "--checkpoint_type", "best", "--output_dir", tmp_path])
        results = json.loads((tmp_path / "single_eval.json").read_text())
        assert "test_acc" in results and "test_f1_macro" in results

    def test_classify_from_official_style_checkpoint_dir_requires_checkpoint(self, tiny_classify_config, tmp_path):
        proc = subprocess.run([sys.executable, str(SCRIPTS / "train_classifier.py"), "--config",
                               str(tiny_classify_config), "--pretrained_path", str(tmp_path / "nothing")],
                              cwd=ROOT, capture_output=True, text=True)
        assert proc.returncode != 0 and "Could not find checkpoint" in proc.stderr


class TestKNN:
    def test_knn_results(self, pretrain_run, tiny_classify_config, tmp_path):
        run([SCRIPTS / "eval_knn.py", "--config", tiny_classify_config, "--pretrained_path", pretrain_run["ckpt"],
             "--k", "5", "--batch_size", "8", "--seeds", "0 1", "--output_base_dir", tmp_path])
        out = tmp_path / "knn" / "pancreatic" / "proto16_koleo0.1_cls1_mc2" / "knn_results.json"
        assert out.exists()
        results = json.loads(out.read_text())
        assert results["seeds"] == [0, 1] and results["k"] == 5
        assert 0 <= results["knn_top1_acc_mean"] <= 100
        assert len(results["per_seed"]) == 2


class TestLowShot:
    def test_lowshot_and_aggregate(self, pretrain_run, tiny_classify_config, tmp_path):
        for ls in (0, 1):
            run([SCRIPTS / "eval_lowshot.py", "--config", tiny_classify_config,
                 "--pretrained_path", pretrain_run["ckpt"], "--k_shot", "2", "--label_seed", str(ls),
                 "--train_seed", "0", "--max_epochs", "1", "--output_dir", tmp_path / "lowshot",
                 "--method_tag", "kodiak", *(["--freeze_backbone"] if ls else [])])
        results = sorted((tmp_path / "lowshot").rglob("test_results.json"))
        assert len(results) == 2
        r = json.loads(results[0].read_text())
        assert r["k_shot"] == 2 and r["n_train_lowshot"] == 6 and r["n_classes"] == 3
        assert 0 <= r["test_acc"] <= 1

        log = run([SCRIPTS / "aggregate_lowshot.py", "--input_dir", tmp_path / "lowshot",
                   "--csv", tmp_path / "agg.csv"])
        assert "2-shot" in log and "kodiak" in log
        assert (tmp_path / "agg.csv").read_text().count("\n") == 3  # header + 2 rows
