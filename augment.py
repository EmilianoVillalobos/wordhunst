import os
import cv2
import numpy as np
from glob import glob
from pathlib import Path
import random

# ==== CONFIG ====

DATASET_DIR = "dataset"     # root with subfolders A, B, C, ...
AUGS_PER_IMAGE = 8          # how many augmented versions to create per original
MAX_ROT_DEG = 8             # max rotation in degrees (±)
MAX_SHIFT_PX = 3            # max translation in pixels (±)
SCALE_JITTER = 0.07         # up/down scaling (e.g. 0.07 = ±7%)
BRIGHTNESS_JITTER = 0.15    # multiplicative factor (1± this)
NOISE_STD = 5.0             # Gaussian noise std dev (0 = none)


# ==== AUGMENTATION FUNCTIONS ====

def random_affine(img):
    """Small rotation + translation + scale."""
    h, w = img.shape[:2]
    center = (w / 2, h / 2)

    # random rotation
    angle = random.uniform(-MAX_ROT_DEG, MAX_ROT_DEG)

    # random scale
    scale = 1.0 + random.uniform(-SCALE_JITTER, SCALE_JITTER)

    # affine rotation+scale
    M = cv2.getRotationMatrix2D(center, angle, scale)

    # random translation
    tx = random.uniform(-MAX_SHIFT_PX, MAX_SHIFT_PX)
    ty = random.uniform(-MAX_SHIFT_PX, MAX_SHIFT_PX)
    M[0, 2] += tx
    M[1, 2] += ty

    augmented = cv2.warpAffine(
        img,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )
    return augmented


def random_brightness_contrast(img):
    """Small brightness/contrast jitter."""
    img = img.astype(np.float32)

    # brightness factor ~ [1-BJ, 1+BJ]
    b_factor = 1.0 + random.uniform(-BRIGHTNESS_JITTER, BRIGHTNESS_JITTER)

    # tiny contrast jitter
    c_factor = 1.0 + random.uniform(-0.08, 0.08)

    mean = img.mean()
    img = (img - mean) * c_factor + mean
    img = img * b_factor

    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def add_noise(img):
    """Add light Gaussian noise."""
    if NOISE_STD <= 0:
        return img
    noise = np.random.normal(0, NOISE_STD, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def augment_once(img):
    """Apply a pipeline of augmentations once."""
    out = img.copy()
    out = random_affine(out)
    out = random_brightness_contrast(out)
    out = add_noise(out)
    return out


# ==== MAIN ====

def augment_dataset(root=DATASET_DIR, augs_per_image=AUGS_PER_IMAGE):
    root_path = Path(root)
    assert root_path.is_dir(), f"{root} is not a directory"

    # iterate over each letter folder (A, B, C, ...)
    letter_dirs = sorted([p for p in root_path.iterdir() if p.is_dir()])

    for letter_dir in letter_dirs:
        letter = letter_dir.name
        print(f"\nProcessing letter '{letter}'")

        # find original PNGs (exclude any already-augmented ones)
        png_paths = sorted(
            p for p in letter_dir.glob("*.png")
            if "_aug" not in p.name
        )

        print(f"  Found {len(png_paths)} base images")

        for img_path in png_paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
            if img is None:
                print(f"  [warn] Could not read {img_path}")
                continue

            stem = img_path.stem  # e.g., 'A_001'

            # how many augmented files already exist for this base?
            existing_augs = glob(str(letter_dir / f"{stem}_aug*.png"))
            start_idx = len(existing_augs) + 1

            for i in range(start_idx, start_idx + augs_per_image):
                aug_img = augment_once(img)
                out_name = f"{stem}_aug{i:02d}.png"
                out_path = letter_dir / out_name
                cv2.imwrite(str(out_path), aug_img)

        print(f"  Done augmenting '{letter}'.")

    print("\nAll done.")


if __name__ == "__main__":
    augment_dataset()
