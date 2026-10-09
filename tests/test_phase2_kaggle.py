"""Phase 2 Kaggle driver: every trained arm Exp 8 evaluates has an adapter Exp 7 trains."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load():
    spec = importlib.util.spec_from_file_location("phase2_kaggle", ROOT / "scripts" / "phase2_kaggle.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_every_evaluated_arm_has_its_adapter():
    pk = _load()
    for n, nab in ((60, 48), (50, 50), (24, 20)):
        jobs, evals = pk.plan("phi3", n, nab, [42, 43, 44], [25, 50])
        trained = {name for name, _ in jobs}
        assert len(trained) == len(jobs)                                  # no adapter trained twice
        for tag, arms, split in evals:
            for arm in arms:
                if arm in ("A", "E"):                                     # untrained arms
                    continue
                assert f"{arm}_phi3_{tag}" in trained, (n, tag, arm)
            assert (split is not None) == tag.endswith("ablation_D6")
        assert sum("E" in arms for _, arms, _ in evals) == 1               # the safety prompt once
        assert all(b < n for b in (25, 50) if b != n and f"C_phi3_n{b}" in trained)   # n-curve only below N
    jobs, _ = pk.plan("phi3", 60, 48, [42, 43], [25])
    b43 = dict(jobs)["B_ext_phi3_n60_s43"]
    assert b43[b43.index("--seed") + 1] == "43" and "--naturalness-csv" not in b43


def test_learning_check_uses_the_final_epoch_only():
    pk = _load()
    lc = {"max_final_epoch_loss": 0.60, "min_final_epoch_reward_accuracy": 0.75}
    learned = {"log_history": [
        {"loss": 0.69, "rewards/accuracies": 0.5, "epoch": 0.5}, {"loss": 0.66, "rewards/accuracies": 0.6, "epoch": 1.0},
        {"loss": 0.55, "rewards/accuracies": 0.8, "epoch": 1.5}, {"loss": 0.50, "rewards/accuracies": 0.9, "epoch": 2.0},
        {"train_loss": 0.6, "epoch": 2.0}]}
    r = pk.learning_check(learned, 2, lc)
    assert r["pass"] and r["n_final_steps"] == 2 and abs(r["final_epoch_loss"] - 0.525) < 1e-9
    flat = {"log_history": [{"loss": 0.69, "rewards/accuracies": 0.5, "epoch": e} for e in (0.5, 1.0, 1.5, 2.0)]}
    assert not pk.learning_check(flat, 2, lc)["pass"]
    assert not pk.learning_check({"log_history": []}, 2, lc)["pass"]       # nothing logged is never a pass


def test_stage_finds_the_bundle_wherever_kaggle_puts_it(tmp_path, monkeypatch):
    import zipfile
    pk = _load()
    src = tmp_path / "src"
    for f in pk.bundle_files():
        (src / f).parent.mkdir(parents=True, exist_ok=True)
        (src / f).write_text("x")
    nested = tmp_path / "input" / "csjail-phase2" / "phase2_upload"      # unpacked one level deeper
    nested.mkdir(parents=True)
    for d in ("outputs", "data"):
        import shutil
        shutil.copytree(src / d, nested / d)
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(pk, "ROOT", repo)
    assert pk.cmd_stage(tmp_path / "input") == 0
    assert all((repo / f).exists() for f in pk.bundle_files())
    repo2, zin = tmp_path / "repo2", tmp_path / "zin" / "any-name"         # the zip itself, any folder name
    repo2.mkdir(); zin.mkdir(parents=True)
    with zipfile.ZipFile(zin / "phase2_upload.zip", "w") as z:
        for f in pk.bundle_files():
            z.write(src / f, f.as_posix())
    monkeypatch.setattr(pk, "ROOT", repo2)
    assert pk.cmd_stage(tmp_path / "zin") == 0 and all((repo2 / f).exists() for f in pk.bundle_files())
    assert pk.cmd_stage(tmp_path / "empty-dir-that-does-not-exist") == 1   # not found -> FAIL, not a crash
