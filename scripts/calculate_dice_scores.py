from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def extract_base_name(filename: str) -> str:
    return re.sub(r"_heatmap$", "", Path(filename).stem)


def load_gray(path: str | Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[WARNING] Failed to read image: {path}")
    return img


def dice_from_binary(bin1: np.ndarray, bin2: np.ndarray) -> float:
    inter = np.logical_and(bin1, bin2).sum()
    sum1 = bin1.sum()
    sum2 = bin2.sum()
    return float((2 * inter / (sum1 + sum2)) if (sum1 + sum2) > 0 else 1.0)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Calculate mean Dice scores between manual annotation masks and "
            "attention heatmaps exported in grayscale mode. "
            "This script assumes that label-dir contains only positive-case "
            "manual annotation masks."
        )
    )
    parser.add_argument(
        "--label-dir",
        type=str,
        required=True,
        help="Folder containing manual annotation masks (positive cases only, binary PNG images).",
    )
    parser.add_argument(
        "--heatmap-dir",
        type=str,
        required=True,
        help="Folder containing grayscale attention heatmaps.",
    )
    parser.add_argument(
        "--output-csv",
        type=str,
        required=True,
        help="Path to save the per-image and mean Dice scores.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=2736,
        help="Target image width used for resizing before Dice calculation.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=1824,
        help="Target image height used for resizing before Dice calculation.",
    )
    parser.add_argument(
        "--threshold-labeled",
        type=float,
        default=128,
        help="Threshold for binarizing manual annotation masks.",
    )
    parser.add_argument(
        "--threshold-mapping",
        type=float,
        default=255 * 0.50,
        help=(
            "Threshold for binarizing grayscale attention heatmaps. "
            "For the default grayscale visualization settings "
            "(center=0.8, scale_factor=30), 255*0.50 is effectively "
            "equivalent to thresholding the underlying normalized attention "
            "scores at 0.8."
        ),
    )
    parser.add_argument(
        "--strict-match",
        action="store_true",
        help=(
            "Raise an error if any label image does not have a matching heatmap. "
            "By default, unmatched files are skipped with a warning."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()

    target_size = (args.width, args.height)

    label_dir = Path(args.label_dir)
    heatmap_dir = Path(args.heatmap_dir)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    if not label_dir.exists():
        raise FileNotFoundError(f"Label directory not found: {label_dir}")
    if not heatmap_dir.exists():
        raise FileNotFoundError(f"Heatmap directory not found: {heatmap_dir}")

    label_files = {
        Path(f).stem: label_dir / f
        for f in os.listdir(label_dir)
        if f.lower().endswith(".png")
    }
    heatmap_files = {
        extract_base_name(f): heatmap_dir / f
        for f in os.listdir(heatmap_dir)
        if f.lower().endswith(".png")
    }

    if not label_files:
        raise RuntimeError("No PNG label files were found in label-dir.")
    if not heatmap_files:
        raise RuntimeError("No PNG heatmap files were found in heatmap-dir.")

    label_keys = set(label_files.keys())
    heatmap_keys = set(heatmap_files.keys())
    common_keys = sorted(label_keys & heatmap_keys)
    missing_heatmaps = sorted(label_keys - heatmap_keys)

    if missing_heatmaps:
        msg = (
            "The following label images do not have matching heatmaps: "
            + ", ".join(missing_heatmaps[:10])
        )
        if len(missing_heatmaps) > 10:
            msg += f" ... ({len(missing_heatmaps)} files total)"
        if args.strict_match:
            raise RuntimeError(msg)
        print(f"[WARNING] {msg}")

    if not common_keys:
        raise RuntimeError("No matching PNG files were found between label-dir and heatmap-dir.")

    results = []

    for key in common_keys:
        path_lbl = label_files[key]
        path_att = heatmap_files[key]

        lbl = load_gray(path_lbl)
        att = load_gray(path_att)
        if lbl is None or att is None:
            continue

        lbl = cv2.resize(lbl, target_size, interpolation=cv2.INTER_NEAREST)
        att = cv2.resize(att, target_size, interpolation=cv2.INTER_LINEAR)

        bin_lbl = lbl >= args.threshold_labeled
        bin_att = att >= args.threshold_mapping

        dice = dice_from_binary(bin_lbl, bin_att)
        results.append({"filename": key, "dice": dice})

    df = pd.DataFrame(results)

    if df.empty:
        raise RuntimeError("No Dice scores were calculated.")

    mean_row = pd.DataFrame([
        {"filename": "MEAN_DICE", "dice": float(df["dice"].mean())}
    ])
    df_out = pd.concat([df, mean_row], ignore_index=True)
    df_out.to_csv(output_csv, index=False, encoding="utf-8-sig")

    print(f"threshold_labeled: {args.threshold_labeled}")
    print(f"threshold_mapping: {args.threshold_mapping}")
    print(df_out)


if __name__ == "__main__":
    main()
