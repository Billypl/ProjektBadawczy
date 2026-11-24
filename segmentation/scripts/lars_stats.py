"""Inspect LaRS: counts, resolutions, mask values, class balance, panoptic categories.

Prints a markdown report (paste into results/notes.md) and writes N random
image+mask overlays to runs/checks/lars/.

    python scripts/lars_stats.py [--root data/raw/lars] [--overlays 20]
"""

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from usvseg.data.lars import DEFAULT_ROOT, Lars
from usvseg.data.taxonomy import SEMANTIC_CLASSES
from usvseg.viz.overlay import overlay, side_by_side

OUT_DIR = Path(__file__).resolve().parents[1] / "runs" / "checks" / "lars"


def split_stats(ds: Lars) -> list[str]:
    resolutions = Counter()
    values = Counter()
    missing = 0
    for s in tqdm(ds.samples, desc=ds.split):
        with Image.open(s.image_path) as im:
            resolutions[im.size] += 1
        if s.semantic_path is None:
            missing += 1
            continue
        mask = s.semantic()
        if mask.shape[::-1] != Image.open(s.image_path).size:
            print(f"size mismatch: {s.name} mask {mask.shape} vs image")
        u, c = np.unique(mask, return_counts=True)
        values.update(dict(zip(u.tolist(), c.tolist())))

    lines = [f"### {ds.split} ({len(ds)} images, {missing} without semantic mask)", ""]
    lines.append("Resolutions (W x H: count): " + ", ".join(
        f"{w}x{h}: {n}" for (w, h), n in resolutions.most_common()))
    if values:
        total = sum(values.values())
        lines.append("")
        lines.append("| mask value | class | pixel share |")
        lines.append("|---|---|---|")
        for v in sorted(values):
            name = SEMANTIC_CLASSES.get(v, "ignore" if v == 255 else "UNKNOWN")
            lines.append(f"| {v} | {name} | {100 * values[v] / total:.2f}% |")
    lines.append("")
    return lines


def panoptic_stats(ds: Lars) -> list[str]:
    try:
        pan = ds.panoptic_annotations
    except FileNotFoundError:
        return [f"(no panoptic_annotations.json for {ds.source_split})", ""]
    seg_counts = Counter()
    for ann in pan["annotations"]:
        for seg in ann["segments_info"]:
            seg_counts[seg["category_id"]] += 1
    lines = [f"### panoptic categories ({ds.source_split})", "",
             "| id | name | isthing | segments |", "|---|---|---|---|"]
    for cat in pan["categories"]:
        lines.append(f"| {cat['id']} | {cat['name']} | {cat.get('isthing', '?')} | {seg_counts[cat['id']]} |")
    lines.append("")
    return lines


def write_overlays(ds: Lars, n: int, seed: int = 0) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    labelled = [s for s in ds.samples if s.semantic_path is not None]
    for s in rng.choice(labelled, size=min(n, len(labelled)), replace=False):
        img = s.image()
        panel = side_by_side([img, overlay(img, s.semantic())], [s.name, "GT semantic"])
        Image.fromarray(panel).save(OUT_DIR / f"{s.name}.jpg", quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--overlays", type=int, default=20)
    args = ap.parse_args()

    report = ["## LaRS dataset statistics", ""]
    for split in ("train", "dev", "val", "test"):
        try:
            report += split_stats(Lars(split, args.root))
        except FileNotFoundError as e:
            report += [f"### {split}: {e}", ""]
    report += panoptic_stats(Lars("train", args.root))
    print("\n".join(report))

    write_overlays(Lars("train", args.root), args.overlays)
    print(f"overlays -> {OUT_DIR}")


if __name__ == "__main__":
    main()
