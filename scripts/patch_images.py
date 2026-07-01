from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2


SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def pad_to_patch_size_with_reflection(image, patch_size):
    """
    Pad the image by reflection so that height and width become
    multiples of patch_size.
    """
    height, width = image.shape[:2]

    new_height = math.ceil(height / patch_size) * patch_size
    new_width = math.ceil(width / patch_size) * patch_size

    pad_h = new_height - height
    pad_w = new_width - width

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    return cv2.copyMakeBorder(
        image,
        top,
        bottom,
        left,
        right,
        borderType=cv2.BORDER_REFLECT,
    )


def split_images_in_folder(
    input_folder,
    output_folder,
    resize_scale=0.5,
    patch_size=256,
    overlap_step=0,
    mirroring=False,
):
    """
    Split all images in input_folder into patches and save them to output_folder.

    Parameters
    ----------
    input_folder : str or Path
        Folder containing input images.
    output_folder : str or Path
        Folder to save patch images.
    resize_scale : float
        Scale factor applied before patch extraction.
    patch_size : int
        Patch size in pixels.
    overlap_step : int
        Overlap size in pixels. The actual stride is patch_size - overlap_step.
    mirroring : bool
        If True, images are padded by reflection so that edge regions are retained.
    """
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if resize_scale <= 0:
        raise ValueError("resize_scale must be > 0.")
    if patch_size <= 0:
        raise ValueError("patch_size must be > 0.")
    if overlap_step < 0:
        raise ValueError("overlap_step must be >= 0.")
    if overlap_step >= patch_size:
        raise ValueError("overlap_step must be smaller than patch_size.")

    step = patch_size - overlap_step

    image_files = sorted(
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not image_files:
        print(f"[WARNING] No supported image files found in: {input_folder}")
        return

    for image_path in image_files:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARNING] Failed to read image: {image_path}")
            continue

        height, width = image.shape[:2]
        resized_width = int(width * resize_scale)
        resized_height = int(height * resize_scale)

        if resized_width <= 0 or resized_height <= 0:
            print(f"[WARNING] Invalid resized shape for image: {image_path}")
            continue

        image = cv2.resize(image, (resized_width, resized_height))

        if mirroring:
            image = pad_to_patch_size_with_reflection(image, patch_size)

        height, width = image.shape[:2]

        if height < patch_size or width < patch_size:
            print(
                f"[WARNING] Image smaller than patch_size after resizing, skipped: "
                f"{image_path.name}"
            )
            continue

        count = 0
        stem = image_path.stem

        for y in range(0, height - patch_size + 1, step):
            for x in range(0, width - patch_size + 1, step):
                patch = image[y:y + patch_size, x:x + patch_size]
                patch_name = f"{stem}_{count}.png"

                save_path = output_folder / patch_name
                success = cv2.imwrite(str(save_path), patch)
                if not success:
                    print(f"[WARNING] Failed to save patch: {save_path}")
                    continue

                count += 1

        print(f"{count} patches saved for {image_path.name}")


def process_all_subfolders(
    input_root,
    output_root,
    patch_size=256,
    overlap_step=0,
    resize_scale=0.5,
    mirroring=False,
):
    """
    Process all subfolders under input_root.
    Each subfolder is treated as one image group, and output patches
    are saved to the corresponding subfolder under output_root.
    """
    input_root = Path(input_root)
    output_root = Path(output_root)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root does not exist: {input_root}")

    output_root.mkdir(parents=True, exist_ok=True)

    subfolders = sorted(p for p in input_root.iterdir() if p.is_dir())

    if not subfolders:
        print(f"[WARNING] No subfolders found in: {input_root}")
        return

    for input_folder in subfolders:
        output_folder = output_root / input_folder.name
        print(f"Processing folder: {input_folder}")

        split_images_in_folder(
            input_folder=input_folder,
            output_folder=output_folder,
            resize_scale=resize_scale,
            patch_size=patch_size,
            overlap_step=overlap_step,
            mirroring=mirroring,
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract patches from images for ABMIL analysis."
    )
    parser.add_argument(
        "--input-root",
        type=str,
        required=True,
        help="Root folder containing subfolders of input images.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Root folder to save extracted patches.",
    )
    parser.add_argument(
        "--resize-scale",
        type=float,
        default=0.5,
        help="Resize scale applied before patch extraction.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=256,
        help="Patch size in pixels.",
    )
    parser.add_argument(
        "--overlap-step",
        type=int,
        default=0,
        help="Overlap size in pixels. Stride = patch_size - overlap_step.",
    )
    parser.add_argument(
        "--mirroring",
        action="store_true",
        help="Pad images by reflection to retain edge regions.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    process_all_subfolders(
        input_root=args.input_root,
        output_root=args.output_root,
        resize_scale=args.resize_scale,
        patch_size=args.patch_size,
        overlap_step=args.overlap_step,
        mirroring=args.mirroring,
    )


if __name__ == "__main__":
    main()