# dss_char_aug/utils.py
import os
import cv2
import numpy as np
from glob import glob


def ensure_dir(path):

    os.makedirs(path, exist_ok=True)


def get_image_paths(folder, ext=".png"):
    """Get sorted list of image paths with specific extension."""
    return sorted(glob(os.path.join(folder, f"*{ext}")))


def load_image(path, grayscale=True):
    """Load an image from disk."""
    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    return cv2.imread(path, flag)


def save_image(img, path):
    """Save an image to disk."""
    cv2.imwrite(path, img)


def show_image(img, title="Image"):
    """Display an image (for debugging)."""
    from matplotlib import pyplot as plt
    plt.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    plt.show()

