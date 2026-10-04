"""Portable OBS crop bank: reuse HybridOpeningMatcher without Linux video paths."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

from weapon_lamp_detect.build_dataset import load_dataset, sample_crop
from weapon_lamp_detect.hybrid_opening_matcher import CONFIGURATIONS, HybridOpeningMatcher
from weapon_lamp_detect.match_data import atomic_json, image_bytes, now, read_json

HERE = Path(__file__).resolve().parent
DEFAULT_BUNDLE = HERE / "data" / "hybrid_candidates"
DEFAULT_EXPERIMENT = HERE.parent / "weapon_lamp_detect" / "data" / "poc_opening_hybrid_20261002"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inside(directory, relative):
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("hybrid見本のpathが保存先の外を参照しています")
    return path


def prepare(experiment, output, kind="full_training"):
    """Export only already selected template sources, never new human labels."""
    experiment, output = Path(experiment).resolve(), Path(output).resolve()
    if kind not in {"independent", "full_training"}:
        raise ValueError("hybrid見本のkindが不正です")
    if output.exists():
        raise ValueError("hybrid見本を上書きしません。新しいoutputを指定してください")
    frozen_path, source_path = experiment / "configuration_frozen.json", experiment / kind / "templates.json"
    frozen, bank = read_json(frozen_path), copy.deepcopy(read_json(source_path))
    configuration = frozen["selected_configuration"]
    if configuration not in CONFIGURATIONS:
        raise ValueError("hybridの固定設定が不正です")
    if not bank.get("templates"):
        raise ValueError("hybrid見本が空です")
    # Preserve the tuned algorithm: don't silently export a changed evaluator.
    for name, expected in frozen.get("algorithm_sha256", {}).items():
        if digest(HERE.parent / "weapon_lamp_detect" / name) != expected:
            raise ValueError(f"hybrid固定後に実装が変わっています: {name}")
    sources = {}
    for entry in bank["templates"]:
        sources.setdefault(entry["dataset"], set()).update(entry["sample_ids"])
    output.mkdir(parents=True)
    files, portable_dirs, sample_count = {}, {}, 0
    for number, (source, ids) in enumerate(sorted(sources.items())):
        relative = f"dataset_{number}"
        directory = output / relative
        directory.mkdir()
        metadata, rows = load_dataset(source)
        by_id = {r["sample_id"]: r for r in rows}
        portable_rows, cache = [], {}
        for i, sample_id in enumerate(sorted(ids)):
            row = by_id[sample_id]
            crop = sample_crop(source, row, metadata, cache)
            crop_name = f"crops/{i:05d}.png"
            path = directory / crop_name
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(image_bytes(crop, ".png"))
            files[f"{relative}/{crop_name}"] = digest(path)
            portable_rows.append({**row, "crop": crop_name, "context": None})
        atomic_json(directory / "dataset.json", {"schema_version": 1, "videos": {},
                    "source_kind": "portable_template_crops", "purpose": "annotation suggestions only, NOT evaluation"})
        (directory / "samples.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in portable_rows), encoding="utf-8")
        for name in ("dataset.json", "samples.jsonl"):
            files[f"{relative}/{name}"] = digest(directory / name)
        portable_dirs[source] = relative
        sample_count += len(portable_rows)
    for entry in bank["templates"]:
        entry["dataset"] = portable_dirs[entry["dataset"]]
    bank["dataset"] = next(iter(portable_dirs.values()))
    bank["purpose"] = "manual annotation suggestions; not an accuracy claim"
    info = {"schema_version": 1, "created_at": now(), "configuration": configuration,
            "source_kind": kind, "source_experiment": str(experiment),
            "source_manifest_sha256": digest(source_path), "source_frozen_sha256": digest(frozen_path),
            "algorithm_sha256": frozen.get("algorithm_sha256", {}),
            "weapon_count": len({e["weapon"] for e in bank["templates"]}),
            "sample_count": sample_count, "bank": bank, "files": files,
            "note": "Frozen OBS hybrid; live labels are NOT automatically templates; current-frame suggestions only"}
    # Ready marker is published last: interrupted preparation cannot be loaded.
    atomic_json(output / "bundle.json", info)
    return info


def bundle_info(directory):
    directory = Path(directory)
    path = directory / "bundle.json"
    if not path.is_file():
        return {"engine": "hybrid", "available": False, "directory": str(directory),
                "notice": "hybrid見本がありません。READMEの見本準備コマンドを実行してください（収集・人力入力は可能です）"}
    try:
        info = read_json(path)
        return {"engine": "hybrid", "available": True, "directory": str(directory),
                "configuration": info["configuration"], "weapon_count": info["weapon_count"],
                "source_kind": info["source_kind"], "single_frame": True,
                "notice": "見本がある武器だけが候補対象です。追加ラベルは自動反映されません"}
    except (OSError, ValueError, KeyError, TypeError):
        return {"engine": "hybrid", "available": False, "directory": str(directory),
                "notice": "hybrid見本のmanifestを読めません。収集・人力入力は継続できます"}


def load(directory):
    directory = Path(directory).resolve()
    if not bundle_info(directory)["available"]:
        raise ValueError("hybrid見本がありません。python -m live_weapon_collect.hybrid_candidates prepare を実行してください。候補なしでも人力入力・収集できます")
    info = read_json(directory / "bundle.json")
    if info.get("schema_version") != 1 or info["configuration"] not in CONFIGURATIONS:
        raise ValueError("hybrid見本の形式・設定が不正です")
    for name, expected in info["files"].items():
        if digest(inside(directory, name)) != expected:
            raise ValueError(f"hybrid見本が変更・破損しています: {name}")
    bank = copy.deepcopy(info["bank"])
    for entry in bank["templates"]:
        entry["dataset"] = str(inside(directory, entry["dataset"]))
    bank["dataset"] = str(inside(directory, bank["dataset"]))
    return HybridOpeningMatcher(bank, info["configuration"])


def main():
    parser = argparse.ArgumentParser(description="既存の固定hybrid見本をWindowsでも読めるcrop bundleへ書き出す")
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    p.add_argument("--output", type=Path, default=DEFAULT_BUNDLE)
    p.add_argument("--kind", choices=["independent", "full_training"], default="full_training")
    a = parser.parse_args()
    info = prepare(a.experiment, a.output, a.kind)
    print(f"hybrid {info['configuration']}: {info['weapon_count']}武器 / {info['sample_count']}見本crop -> {a.output}")


if __name__ == "__main__":
    main()
