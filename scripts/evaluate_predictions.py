#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluate binary classification predictions using:
- a threshold determined from the primary OOF predictions
- ROC curves for the primary and optional extra test predictions
- a confusion matrix for the primary test predictions

Expected CSV columns
--------------------
Ground truth:
- target, y_true, or label

Predictions:
- pred_0 and pred_1
  or
- logit_0 and logit_1

If pred_0 + pred_1 is not approximately 1, the script assumes the two
prediction columns are logits and converts them to probabilities using softmax.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

mpl.rcParams["axes.unicode_minus"] = False


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def set_font_family(font_family: str | None) -> None:
    if font_family:
        mpl.rcParams["font.family"] = font_family


def load_as_probs(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(csv_path)
    cols = {c.lower(): c for c in df.columns}

    y_col = cols.get("target") or cols.get("y_true") or cols.get("label")
    p0_col = cols.get("pred_0") or cols.get("p0") or cols.get("logit_0")
    p1_col = cols.get("pred_1") or cols.get("p1") or cols.get("logit_1")

    if y_col is None or p0_col is None or p1_col is None:
        raise ValueError(
            f"[{csv_path.name}] Missing required columns. "
            "Expected one of [target, y_true, label] and "
            "one of [pred_0, p0, logit_0] / [pred_1, p1, logit_1]."
        )

    y = df[y_col].astype(int).to_numpy()
    p0 = df[p0_col].astype(float).to_numpy()
    p1 = df[p1_col].astype(float).to_numpy()

    s = p0 + p1
    is_prob_like = (abs(s.mean() - 1.0) < 1e-3) and (np.abs(s - 1.0).mean() < 1e-3)

    if not is_prob_like:
        logits = np.vstack([p0, p1]).T
        logits = logits - logits.max(axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)
        p1 = probs[:, 1]

    return y, p1


def choose_threshold(
    y: np.ndarray,
    p: np.ndarray,
    mode: str = "youden",
    fixed_thr: float = 0.5,
    target: float = 0.9,
) -> float:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)

    if mode == "fixed":
        return float(fixed_thr)

    fpr, tpr, thr = roc_curve(y, p)

    # Remove the first threshold (inf)
    fpr, tpr, thr = fpr[1:], tpr[1:], thr[1:]

    if len(thr) == 0:
        return float(fixed_thr)

    if mode == "youden":
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])

    if mode == "sens_at":
        return float(thr[int(np.argmin(np.abs(tpr - target)))])

    if mode == "spec_at":
        spec = 1.0 - fpr
        return float(thr[int(np.argmin(np.abs(spec - target)))])

    return float(fixed_thr)


def binary_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)
    yhat = (p >= thr).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    eps = 1e-12

    sens = tp / (tp + fn + eps)
    spec = tn / (tn + fp + eps)
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    ppv = tp / (tp + fp + eps)
    npv = tn / (tn + fn + eps)
    f1 = 2 * ppv * sens / (ppv + sens + eps)
    bacc = 0.5 * (sens + spec)

    return {
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
        "precision": ppv,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": bacc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def derive_confusion_matrix_path(out_png: Path, out_cm_png: str | None) -> Path:
    if out_cm_png:
        return Path(out_cm_png)
    return out_png.with_suffix("") .with_name(out_png.with_suffix("").name + "_cm").with_suffix(".png")


def save_confusion_matrix(
    cm: np.ndarray,
    out_path: Path,
    row_labels: list[str],
    col_labels: list[str],
    cmap: str = "Blues",
) -> None:
    ensure_parent_dir(out_path)

    # 行方向に正規化（色用）
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(3.5, 3), dpi=300)

    # 色は相対値
    im = ax.imshow(cm_norm, cmap=cmap, vmin=0, vmax=1)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_ticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    cbar.set_ticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0"])
    cbar.set_label("Row-normalized value", fontsize=8)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=9)
    ax.set_yticklabels(row_labels, fontsize=9)

    ax.set_xlabel("Predicted class", fontsize=11, labelpad=6)
    ax.set_ylabel("True class", fontsize=11, rotation=90, labelpad=8, va="center")

    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # 文字色の閾値は相対値ベース
    value_threshold = 0.5

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])  # セル内は絶対値
            ax.text(
                j,
                i,
                str(value),
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > value_threshold else "black",
                fontsize=10,
                fontweight="bold" if i == j else "normal",  # 対角成分だけ太字
            )

    plt.setp(ax.get_yticklabels(), rotation=90, va="center")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def shorten_curve_name(path: Path) -> str:
    match = re.search(r"seed(\d+)", path.name)
    return f"seed{match.group(1)}" if match else path.stem


def save_roc_plot(
    fpr_primary: np.ndarray,
    tpr_primary: np.ndarray,
    auc_primary: float,
    extra_curves: Iterable[tuple[str, np.ndarray, np.ndarray, float]],
    out_path: Path,
    title: str,
    pad: float,
    show_legend: bool,
) -> None:
    ensure_parent_dir(out_path)

    plt.figure(figsize=(3.1, 3.0), dpi=600)
    plt.plot([0, 1], [0, 1], linestyle=":", lw=1, color="#888888", label="chance")
    plt.plot(
        fpr_primary,
        tpr_primary,
        lw=1.5,
        color="#000000",
        label=f"Primary (AUC={auc_primary:.4f})",
    )

    for name, fpr, tpr, auc in extra_curves:
        plt.plot(
            fpr,
            tpr,
            lw=1.0,
            linestyle="--",
            color="#000000",
            label=f"{name} (AUC={auc:.4f})",
        )

    plt.xlim(-pad, 1 + pad)
    plt.ylim(-pad, 1 + pad)
    plt.xticks(np.arange(0, 1.01, 0.2), fontsize=7)
    plt.yticks(np.arange(0, 1.01, 0.2), fontsize=7)
    plt.xlabel("False positive rate", fontsize=9)
    plt.ylabel("True positive rate", fontsize=9)
    plt.title(title, fontsize=12)

    if show_legend:
        plt.legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate binary classification predictions from OOF and test CSV files."
    )
    parser.add_argument("--primary-oof", "--primary_oof", dest="primary_oof", type=str, required=True,
                        help="OOF CSV for the primary seed.")
    parser.add_argument("--primary-test", "--primary_test", dest="primary_test", type=str, required=True,
                        help="Test CSV for the primary seed.")
    parser.add_argument("--extra-tests", "--extra_tests", dest="extra_tests", nargs="*", default=[],
                        help="Optional test CSV files for additional seeds.")
    parser.add_argument("--thr-mode", "--thr_mode", dest="thr_mode", default="youden",
                        choices=["fixed", "youden", "sens_at", "spec_at"],
                        help="Threshold selection method.")
    parser.add_argument("--thr", type=float, default=0.5,
                        help="Fixed threshold used when --thr-mode fixed is selected.")
    parser.add_argument("--target", type=float, default=0.9,
                        help="Target sensitivity or specificity used for sens_at/spec_at.")
    parser.add_argument("--out-png", "--out_png", dest="out_png", default="roc_primary_vs_extra.png",
                        help="Path to save the ROC figure.")
    parser.add_argument("--out-cm-png", "--out_cm_png", dest="out_cm_png", default=None,
                        help="Optional path to save the confusion matrix image. "
                             "If omitted, a file ending with _cm.png is created next to --out-png.")
    parser.add_argument("--out-metrics-csv", "--out_metrics_csv", dest="out_metrics_csv",
                        default="primary_metrics.csv",
                        help="Path to save the primary test metrics CSV.")
    parser.add_argument("--title", type=str, default="ROC (primary solid, extras dashed)",
                        help="Title of the ROC figure.")
    parser.add_argument("--pad", type=float, default=0.02,
                        help="Axis padding for the ROC plot.")
    parser.add_argument("--positive-class-name", default="Cancer",
                        help="Display name for the positive class in the confusion matrix.")
    parser.add_argument("--negative-class-name", default="Non-cancer",
                        help="Display name for the negative class in the confusion matrix.")
    parser.add_argument("--font-family", default="Arial",
                        help="Optional matplotlib font family (for example, Arial).")
    parser.add_argument("--show-legend", action="store_true",
                        help="Show the legend on the ROC plot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_font_family(args.font_family)

    primary_oof_path = Path(args.primary_oof)
    primary_test_path = Path(args.primary_test)
    out_png_path = Path(args.out_png)
    out_metrics_csv_path = Path(args.out_metrics_csv)
    out_cm_png_path = derive_confusion_matrix_path(out_png_path, args.out_cm_png)

    y_oof, p_oof = load_as_probs(primary_oof_path)
    thr = choose_threshold(
        y_oof,
        p_oof,
        mode=args.thr_mode,
        fixed_thr=args.thr,
        target=args.target,
    )

    y_primary, p_primary = load_as_probs(primary_test_path)
    fpr_primary, tpr_primary, _ = roc_curve(y_primary, p_primary)
    auc_primary = roc_auc_score(y_primary, p_primary)

    metrics = binary_metrics(y_primary, p_primary, thr)
    ensure_parent_dir(out_metrics_csv_path)
    pd.DataFrame(
        [{**metrics, "threshold": thr, "AUC": auc_primary, "N": len(y_primary)}]
    ).to_csv(out_metrics_csv_path, index=False)

    print(
        f"[INFO] primary AUC={auc_primary:.6f}, thr={thr:.4f}, "
        f"Sens={metrics['sensitivity']:.3f}, "
        f"Spec={metrics['specificity']:.3f}, "
        f"Acc={metrics['accuracy']:.3f}"
    )

    yhat_primary = (p_primary >= thr).astype(int)
    cm = confusion_matrix(y_primary, yhat_primary, labels=[1, 0])

    row_labels = [args.positive_class_name, args.negative_class_name]
    col_labels = [args.positive_class_name, args.negative_class_name]
    save_confusion_matrix(
        cm=cm,
        out_path=out_cm_png_path,
        row_labels=row_labels,
        col_labels=col_labels,
    )

    extra_curves: list[tuple[str, np.ndarray, np.ndarray, float]] = []
    for path_str in args.extra_tests:
        path = Path(path_str)
        y_extra, p_extra = load_as_probs(path)
        fpr_extra, tpr_extra, _ = roc_curve(y_extra, p_extra)
        auc_extra = roc_auc_score(y_extra, p_extra)
        short_name = shorten_curve_name(path)
        extra_curves.append((short_name, fpr_extra, tpr_extra, auc_extra))
        print(f"[INFO] extra {path.name}: AUC={auc_extra:.6f}, N={len(y_extra)}")

    save_roc_plot(
        fpr_primary=fpr_primary,
        tpr_primary=tpr_primary,
        auc_primary=auc_primary,
        extra_curves=extra_curves,
        out_path=out_png_path,
        title=args.title,
        pad=float(args.pad),
        show_legend=args.show_legend,
    )

    print(f"[SAVE] ROC figure: {out_png_path.resolve()}")
    print(f"[SAVE] Confusion matrix: {out_cm_png_path.resolve()}")
    print(f"[SAVE] Metrics CSV: {out_metrics_csv_path.resolve()}")


if __name__ == "__main__":
    main()
