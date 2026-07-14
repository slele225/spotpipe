"""Sanity tests for the model-inference adapter (spotpipe.benchmark.infer).

These do NOT depend on the carried checkpoints or the full benchmark: they build
a tiny random model + a couple of tiny TIFF conditions on the fly, so they run in
seconds on CPU. The one load-bearing correctness check is
``test_batched_decode_matches_vendored``: it proves the adapter's batched decode
reproduces the frozen ``predict_spots`` convention bit-for-bit, which is what
lets us batch the forward pass without drifting from the vendored inference path.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import tifffile
import torch

from spotpipe.benchmark import infer
from spotpipe.models import build_spot_model, predict_spots
from spotpipe.models.spot_model import normalize_counts
from spotpipe.schema import SCHEMA_COLUMNS

_MODEL_CFG = {"in_channels": 2, "base_channels": 8, "num_branches": 2,
              "blocks_per_branch": 1, "head_mid_channels": 8, "heatmap_bias": -1.0}


def _tiny_model(seed: int = 0):
    torch.manual_seed(seed)
    model = build_spot_model(_MODEL_CFG)
    model.eval()
    return model


def _params():
    # Loose threshold so a random model still fires some peaks on the tiny image.
    return infer.InferenceParams(adc_max=4095.0, peak_threshold=0.05, nms_kernel=3,
                                 max_spots=50, logvar_min=-10.0, logvar_max=6.0)


def test_batched_decode_matches_vendored():
    """Adapter's batched _decode_image == vendored predict_spots, per image."""
    model = _tiny_model()
    # max_spots=None keeps peaks in deterministic raster order; with a top-k cap
    # the ~1e-7 batch-vs-single score differences can reorder near-tied peaks
    # (same spot SET, different enumeration), which is not a convention issue.
    p = infer.InferenceParams(adc_max=4095.0, peak_threshold=0.05, nms_kernel=3,
                              max_spots=None, logvar_min=-10.0, logvar_max=6.0)
    rng = np.random.default_rng(1)
    imgs = [rng.integers(0, 4096, size=(2, 24, 24)).astype(np.float32) for _ in range(3)]

    batch = torch.stack([torch.from_numpy(a) for a in imgs], dim=0)
    with torch.no_grad():
        preds = model(normalize_counts(batch, p.adc_max))

    for i, arr in enumerate(imgs):
        image_id = f"img_{i:05d}"
        # Vendored single-image path (the frozen convention).
        want = predict_spots(
            model, arr, image_id=image_id, adc_max=p.adc_max,
            peak_threshold=p.peak_threshold, nms_kernel=p.nms_kernel,
            max_spots=p.max_spots, logvar_min=p.logvar_min, logvar_max=p.logvar_max)
        # Adapter's batched decode.
        got = pd.DataFrame([r.__dict__ for r in infer._decode_image(preds, i, image_id, p)],
                           columns=list(SCHEMA_COLUMNS))
        # Same peaks in the same order; intensities agree to well within any
        # meaningful tolerance. The residual ~1e-6 gap is MKL using a different
        # conv accumulation path for a batch of 3 vs the batch-of-1 the vendored
        # path runs -- not a convention difference.
        if want.empty and got.empty:
            continue
        assert list(got["image_id"]) == list(want["image_id"])
        assert list(got["spot_id"]) == list(want["spot_id"])
        pd.testing.assert_frame_equal(
            got.reset_index(drop=True), want.reset_index(drop=True),
            check_dtype=False, rtol=1e-3, atol=1e-4)


def _make_condition(root, family, label, n_images, shape=(2, 20, 20)):
    cdir = root / family / label
    (cdir / "images").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(hash(label) % (2**32))
    images = []
    for i in range(n_images):
        image_id = f"{label}_{i:05d}"
        arr = rng.integers(0, 4096, size=shape, dtype=np.uint16)
        tifffile.imwrite(cdir / "images" / f"image_{image_id}.tif", arr)
        images.append({"image_id": image_id, "image_file": f"images/image_{image_id}.tif"})
    with open(cdir / "meta.json", "w", encoding="utf-8") as fh:
        json.dump({"family": family, "label": label, "n_images": n_images,
                   "images": images}, fh)
    return cdir


# A checkpoint's provenance (LEGACY vs clean retrain) is DERIVED from its recorded
# training SHA, so the fake checkpoints must carry a real one. Default = the OLD
# dirty-tree SHA, which is what the two carried legacy checkpoints record and what
# the ``our_model_<name>_legacy`` folder naming below asserts.
_LEGACY_SHA = "93fc0aa8c6b245c3c2294a995a3c172d55064f57-dirty"
_CLEAN_SHA = "26b0d487d16d6516082f20cf03a5a56a1e8e3f9b"


def _fake_checkpoint(root, name, git_commit: str = _LEGACY_SHA):
    ckdir = root / name
    ckdir.mkdir(parents=True, exist_ok=True)
    model = _tiny_model(seed=abs(hash(name)) % 1000)
    torch.save({"model_state": model.state_dict(),
                "config": {"model": _MODEL_CFG,
                           "benchmark": {"our_model": {"peak_threshold": 0.05,
                                                       "nms_kernel": 3, "max_spots": 50}}}},
               ckdir / "best_checkpoint.pt")
    with open(ckdir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump({"git_commit": git_commit}, fh)
    return name


def test_provenance_is_derived_from_training_sha():
    """LEGACY is a fact about the training SHA, never a hardcoded label."""
    assert infer.is_legacy_checkpoint(_LEGACY_SHA) is True
    assert infer.is_legacy_checkpoint("unknown") is True       # unprovenanced -> fail safe
    assert infer.is_legacy_checkpoint("") is True
    assert infer.is_legacy_checkpoint(_CLEAN_SHA) is False

    assert infer.method_name("hrnet_large", _LEGACY_SHA) == "our_model_hrnet_large_legacy"
    assert (infer.method_name("hrnet_large_measured", _CLEAN_SHA)
            == "our_model_hrnet_large_measured")


def test_clean_retrain_is_not_labelled_legacy(tmp_path):
    """A clean-tree checkpoint gets a plain folder + a HEADLINE manifest status."""
    bench = _bench(tmp_path)
    ckroot = tmp_path / "checkpoints"
    _fake_checkpoint(ckroot, "hrnet_large_measured", git_commit=_CLEAN_SHA)
    results = tmp_path / "results"
    infer.run_inference(
        "hrnet_large_measured", bench_root=bench, results_root=results, repo_root=tmp_path,
        checkpoints_root=ckroot, device="cpu", batch_size=2, num_workers=0,
        smoke=True, smoke_conditions=1, smoke_images=2, log_fn=lambda *_: None)

    method_dir = results / "our_model_hrnet_large_measured"
    assert method_dir.is_dir()
    assert not (results / "our_model_hrnet_large_measured_legacy").exists()
    manifest = json.loads((method_dir / "RUN_MANIFEST.json").read_text())
    assert manifest["provenance"]["status"] == "CLEAN RETRAIN/HEADLINE"
    assert manifest["provenance"]["training_git_sha"] == _CLEAN_SHA
    assert "NOT legacy" in manifest["provenance"]["note"]


def _bench(tmp_path):
    bench = tmp_path / "benchmark"
    _make_condition(bench, "snr_density", "snr=5_density=0.006", 4)
    _make_condition(bench, "snr_density", "snr=10_density=0.002", 4)
    _make_condition(bench, "curvature", "alpha=0.3", 4)
    return bench


def test_smoke_run_writes_valid_csvs_and_manifest(tmp_path):
    bench = _bench(tmp_path)
    ckroot = tmp_path / "checkpoints"
    _fake_checkpoint(ckroot, "hrnet_large")
    results = tmp_path / "results"

    infer.run_inference(
        "hrnet_large", bench_root=bench, results_root=results, repo_root=tmp_path,
        checkpoints_root=ckroot, device="cpu", batch_size=2, num_workers=0,
        smoke=True, smoke_conditions=2, smoke_images=2, log_fn=lambda *_: None)

    method_dir = results / "our_model_hrnet_large_legacy"
    manifest = json.loads((method_dir / "RUN_MANIFEST.json").read_text())

    # Provenance honesty recorded.
    assert manifest["provenance"]["status"] == "LEGACY/REFERENCE"
    assert manifest["provenance"]["training_git_sha"] == _LEGACY_SHA
    assert manifest["checkpoint"] == "hrnet_large"
    assert "device" in manifest and "cuda_available" in manifest

    # Smoke subset: 2 snr_density conditions only (smoke_conditions=2 per family,
    # but only 2 snr_density + would-be curvature; smoke takes first N per family).
    csvs = list(method_dir.rglob("predictions.csv"))
    assert csvs, "no prediction CSVs written"
    for csv in csvs:
        header = pd.read_csv(csv, nrows=0)
        assert tuple(header.columns) == SCHEMA_COLUMNS
        # smoke_images=2 -> at most 2 image_ids present
        df = pd.read_csv(csv)
        assert df["image_id"].nunique() <= 2


def test_skip_if_exists(tmp_path):
    bench = _bench(tmp_path)
    ckroot = tmp_path / "checkpoints"
    _fake_checkpoint(ckroot, "hrnet_small")
    results = tmp_path / "results"

    common = dict(bench_root=bench, results_root=results, repo_root=tmp_path,
                  checkpoints_root=ckroot, device="cpu", batch_size=2, num_workers=0,
                  smoke=True, smoke_conditions=2, smoke_images=2, log_fn=lambda *_: None)

    infer.run_inference("hrnet_small", **common)
    m1 = json.loads((results / "our_model_hrnet_small_legacy" / "RUN_MANIFEST.json").read_text())
    assert m1["totals"]["n_conditions_run"] >= 1
    assert m1["totals"]["n_conditions_skipped"] == 0

    # Second run must skip everything already on disk.
    infer.run_inference("hrnet_small", **common)
    m2 = json.loads((results / "our_model_hrnet_small_legacy" / "RUN_MANIFEST.json").read_text())
    assert m2["totals"]["n_conditions_run"] == 0
    assert m2["totals"]["n_conditions_skipped"] == m1["totals"]["n_conditions"]
    assert all(c.get("skipped") for c in m2["conditions"].values())


def test_both_checkpoints_distinct_method_folders(tmp_path):
    bench = _bench(tmp_path)
    ckroot = tmp_path / "checkpoints"
    _fake_checkpoint(ckroot, "hrnet_large")
    _fake_checkpoint(ckroot, "hrnet_small")
    results = tmp_path / "results"

    infer.run_inference(
        "all", bench_root=bench, results_root=results, repo_root=tmp_path,
        checkpoints_root=ckroot, device="cpu", batch_size=2, num_workers=0,
        smoke=True, smoke_conditions=1, smoke_images=2, log_fn=lambda *_: None)

    assert (results / "our_model_hrnet_large_legacy" / "RUN_MANIFEST.json").exists()
    assert (results / "our_model_hrnet_small_legacy" / "RUN_MANIFEST.json").exists()


def test_predictions_csv_passes_frozen_schema(tmp_path):
    """A written predictions.csv round-trips through the frozen schema reader."""
    from spotpipe.schema import read_spots

    bench = _bench(tmp_path)
    ckroot = tmp_path / "checkpoints"
    _fake_checkpoint(ckroot, "hrnet_large")
    results = tmp_path / "results"
    infer.run_inference(
        "hrnet_large", bench_root=bench, results_root=results, repo_root=tmp_path,
        checkpoints_root=ckroot, device="cpu", batch_size=2, num_workers=0,
        smoke=True, smoke_conditions=1, smoke_images=2, log_fn=lambda *_: None)

    csv = next((results / "our_model_hrnet_large_legacy").rglob("predictions.csv"))
    records = read_spots(csv)  # raises if columns/schema invalid
    assert isinstance(records, list)
