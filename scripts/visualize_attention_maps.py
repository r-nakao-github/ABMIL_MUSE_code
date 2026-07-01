from __future__ import annotations

import argparse
import glob
import math
import os
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable, List, Sequence

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from train_abmil_model import AttentionMIL, TestDataset, get_transforms

COLOR_CHOICES = ["inferno", "grayscale"]


def dict_to_namespace(obj):
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [dict_to_namespace(v) for v in obj]
    return obj


def load_config(config_path: str | Path):
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return dict_to_namespace(data)


def seed_torch(seed: int = 42) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def compute_attention_raw(model: AttentionMIL, bag: torch.Tensor) -> np.ndarray:
    """
    Compute raw attention scores for one bag using a trained Attention MIL model.
    """
    model.eval()
    with torch.no_grad():
        bag = bag.to(next(model.parameters()).device)

        if bag.ndim == 5:
            bag = bag.squeeze(0)

        _, c, h, w = bag.shape
        bag = bag.view(-1, c, h, w)

        features = model.model.forward_features(bag)

        if features.dim() == 4 and features.shape[-1] == model.classifier.in_features:
            features = features.permute(0, 3, 1, 2)

        features = F.adaptive_avg_pool2d(features, 1).flatten(1)
        x = features.unsqueeze(0)

        a_v = model.attention_V(x)
        a_u = model.attention_U(x)
        a = model.attention_w(a_v * a_u)
        return a.cpu().numpy().squeeze()


def pad_image_with_reflection(image: np.ndarray, patch_size: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    new_h = math.ceil(h / patch_size) * patch_size
    new_w = math.ceil(w / patch_size) * patch_size
    pad_h = int((new_h - h) / 2)
    pad_w = int((new_w - w) / 2)

    padded = cv2.copyMakeBorder(
        image,
        pad_h,
        pad_h,
        pad_w,
        pad_w,
        borderType=cv2.BORDER_REFLECT,
    )
    return padded, pad_h, pad_w


def generate_attention_heatmap(
    image_path: str | Path,
    attention_scores: np.ndarray,
    patch_size: int,
    stride: int,
    image_resize: tuple[int, int],
    color: str,
    mirroring: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Generate overlay and heatmap images from patch-level attention scores.
    image_resize should be given as (width, height).
    """
    image_path = str(image_path)
    width, height = image_resize

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"[WARNING] Image not found: {image_path}")
        return None, None

    image = cv2.resize(original_image, (width, height))

    pad_h = 0
    pad_w = 0
    if mirroring:
        image, pad_h, pad_w = pad_image_with_reflection(image, patch_size)

    height, width = image.shape[:2]
    overlay = image.copy()
    heatmap_overlay = np.zeros((height, width, 3), dtype=np.uint8)
    grayscale_heatmap = np.zeros((height, width), dtype=np.float32)

    num_patches_x = (width - patch_size) // stride + 1
    num_patches_y = (height - patch_size) // stride + 1

    idx = 0
    for y in range(num_patches_y):
        for x in range(num_patches_x):
            if idx >= len(attention_scores):
                break

            x_pos = x * stride
            y_pos = y * stride
            attn_value = float(attention_scores[idx])

            if color == "inferno":
                heatmap_patch = np.full((patch_size, patch_size), attn_value, dtype=np.float32)
                heatmap_patch = np.uint8(np.clip(255 * heatmap_patch, 0, 255))
                heatmap_patch = cv2.applyColorMap(heatmap_patch, cv2.COLORMAP_INFERNO)
                heatmap_overlay[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size] = heatmap_patch

            elif color == "grayscale":
                alpha = attn_value * 0.5
                white_patch = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
                original_patch = overlay[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size]
                blended_patch = cv2.addWeighted(original_patch, 1 - alpha, white_patch, alpha, 0)
                overlay[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size] = blended_patch
                grayscale_heatmap[y_pos:y_pos + patch_size, x_pos:x_pos + patch_size] = attn_value

            idx += 1

    if color == "grayscale":
        heatmap_output = np.uint8(np.clip(255 * grayscale_heatmap, 0, 255))
        if mirroring and pad_h > 0 and pad_w > 0:
            overlay = overlay[pad_h:-pad_h, pad_w:-pad_w, :]
            heatmap_output = heatmap_output[pad_h:-pad_h, pad_w:-pad_w]
    else:
        heatmap_output = heatmap_overlay.copy()
        overlay = cv2.addWeighted(image, 0.5, heatmap_overlay, 0.5, 0)
        if mirroring and pad_h > 0 and pad_w > 0:
            overlay = overlay[pad_h:-pad_h, pad_w:-pad_w, :]
            heatmap_output = heatmap_output[pad_h:-pad_h, pad_w:-pad_w, :]

    return overlay, heatmap_output


def normalize_attention(
    attention: np.ndarray,
    global_min: float,
    global_max: float,
    color: str,
    center: float,
    scale_factor: float,
) -> np.ndarray:
    denom = global_max - global_min
    if denom <= 0:
        normalized = np.zeros_like(attention, dtype=np.float32)
    else:
        normalized = (attention - global_min) / denom

    if color == "inferno":
        return normalized

    if color == "grayscale":
        shifted = (normalized - center) * scale_factor
        return torch.sigmoid(torch.tensor(shifted, dtype=torch.float32)).numpy()

    raise ValueError(f"Unsupported color mode: {color}")


def get_model_paths(experiment_dir: str | Path, model_pattern: str) -> List[str]:
    pattern = os.path.join(str(experiment_dir), model_pattern)
    model_paths = sorted(glob.glob(pattern))
    if not model_paths:
        raise FileNotFoundError(f"No model files matched: {pattern}")
    return model_paths


def load_models(cfg, model_paths: Sequence[str], device: torch.device) -> list[AttentionMIL]:
    models = []
    for path in model_paths:
        model = AttentionMIL(cfg, base_model=cfg.model.name, pretrain=False).to(device)
        model.load_state_dict(torch.load(path, map_location=device))
        model.eval()
        models.append(model)
    return models


def collect_global_min_max(
    image_ids: Iterable[str],
    dataset: TestDataset,
    models: Sequence[AttentionMIL],
) -> tuple[float, float]:
    global_min = float("inf")
    global_max = float("-inf")

    for image_id in image_ids:
        bag = dataset.get_by_id(image_id)
        attention_scores_list = []
        for model in models:
            attention_scores_list.append(compute_attention_raw(model, bag.unsqueeze(0)))

        mean_attention_scores = np.mean(np.stack(attention_scores_list), axis=0)
        global_min = min(global_min, float(mean_attention_scores.min()))
        global_max = max(global_max, float(mean_attention_scores.max()))

    return global_min, global_max


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate attention heatmaps from trained ABMIL models."
    )
    parser.add_argument("--test-csv", type=str, required=True, help="Path to the test metadata CSV.")
    parser.add_argument(
        "--image-folder",
        type=str,
        required=True,
        help="Folder containing the original test images used for visualization.",
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Experiment directory containing trained model weights and the saved config file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for heatmaps. Default: <experiment-dir>/attention_maps",
    )
    parser.add_argument(
        "--config-name",
        type=str,
        default="config_used_exp0.yaml",
        help="Configuration filename saved in the experiment directory.",
    )
    parser.add_argument(
        "--model-pattern",
        type=str,
        default="fold*_0_best_auc_loss.pth",
        help="Glob pattern for trained model files inside the experiment directory.",
    )
    parser.add_argument("--image-width", type=int, default=2736, help="Resized image width.")
    parser.add_argument("--image-height", type=int, default=1824, help="Resized image height.")
    parser.add_argument("--patch-size", type=int, default=224, help="Patch size in pixels.")
    parser.add_argument("--stride", type=int, default=224, help="Stride in pixels.")
    parser.add_argument(
        "--color",
        type=str,
        default="inferno",
        choices=COLOR_CHOICES,
        help="Color mode for heatmap generation: inferno or grayscale.",
    )
    parser.add_argument(
        "--center",
        type=float,
        default=0.80,
        help="Center value used only for grayscale mode normalization.",
    )
    parser.add_argument(
        "--scale-factor",
        type=float,
        default=30.0,
        help="Scale factor used only for grayscale mode normalization.",
    )
    parser.add_argument(
        "--image-ext",
        type=str,
        default=".png",
        help="Extension of the original images in the image folder.",
    )
    parser.add_argument("--mirroring", action="store_true", help="Apply reflection padding before mapping.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    args.color = args.color.lower()

    seed_torch(args.seed)

    experiment_dir = Path(args.experiment_dir)
    output_dir = Path(args.output_dir) if args.output_dir else experiment_dir / "attention_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(experiment_dir / args.config_name)
    df_test = pd.read_csv(args.test_csv)
    test_dataset = TestDataset(df_test, cfg, transform1=get_transforms(data="test", CFG=cfg))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_paths = get_model_paths(experiment_dir, args.model_pattern)
    models = load_models(cfg, model_paths, device)

    image_ids = df_test["ID"].tolist()
    global_min, global_max = collect_global_min_max(image_ids, test_dataset, models)
    print(f"global_min: {global_min}, global_max: {global_max}")

    for image_id in image_ids:
        bag = test_dataset.get_by_id(image_id)

        attention_scores_list = []
        for model in models:
            attention_scores_list.append(compute_attention_raw(model, bag.unsqueeze(0)))

        mean_attention_scores = np.mean(np.stack(attention_scores_list), axis=0)
        normalized_attention_scores = normalize_attention(
            mean_attention_scores,
            global_min=global_min,
            global_max=global_max,
            color=args.color,
            center=args.center,
            scale_factor=args.scale_factor,
        )

        image_path = Path(args.image_folder) / f"{image_id}{args.image_ext}"
        overlay, heatmap = generate_attention_heatmap(
            image_path=image_path,
            attention_scores=normalized_attention_scores,
            patch_size=args.patch_size,
            stride=args.stride,
            image_resize=(args.image_width, args.image_height),
            color=args.color,
            mirroring=args.mirroring,
        )

        if overlay is None or heatmap is None:
            continue

        heatmap_path = output_dir / f"{image_id}_heatmap.png"
        overlay_path = output_dir / f"{image_id}_image+heatmap.png"
        cv2.imwrite(str(heatmap_path), heatmap)
        cv2.imwrite(str(overlay_path), overlay)
        print(f"Saved: {heatmap_path}")
        print(f"Saved: {overlay_path}")


if __name__ == "__main__":
    main()
