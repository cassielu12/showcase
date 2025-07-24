#this is modified from https://github.com/GrHound/imagemorph.c/blob/master/imagemorph.py
import math
import random
import numpy as np
import cv2
import math
import random
import numpy as np
import cv2
from .base import BaseTransform

class Distortion(BaseTransform):
    def __init__(self, amp=2.0, sigma=30.0, mode='random'):
        super().__init__(mode)
        self.amp = amp
        self.sigma = sigma

    def __call__(self, image: np.ndarray, mask: np.ndarray = None):
        if image is None or image.ndim != 2:
            raise ValueError("Input must be a 2D grayscale image.")

        h, w = image.shape[:2]

        # Randomized or fixed parameters
        if self.mode == 'random':
            amp = random.uniform(0.0, self.amp)
            sigma = random.uniform(1e-5, self.sigma)
        else:
            amp = self.amp
            sigma = self.sigma

        if sigma > h / 2.5 or sigma > w / 2.5:
            raise ValueError("Gaussian smoothing kernel too large.")

        # Generate distortion field
        d_x = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        d_y = np.random.uniform(-1.0, 1.0, (h, w)).astype(np.float32)
        d_x = cv2.GaussianBlur(d_x, (0, 0), sigma)
        d_y = cv2.GaussianBlur(d_y, (0, 0), sigma)
        avg = np.mean(np.sqrt(d_x**2 + d_y**2))
        d_x = amp * d_x / avg
        d_y = amp * d_y / avg

        map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
        map_x = (map_x + d_x).astype(np.float32)
        map_y = (map_y + d_y).astype(np.float32)

        # Distort image (grayscale)
        distorted_img = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

        if mask is not None:
            distorted_mask = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
            return distorted_img, distorted_mask

        return distorted_img
