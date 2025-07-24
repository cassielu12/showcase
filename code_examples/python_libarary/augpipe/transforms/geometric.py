
import cv2
import numpy as np
import random
from .base import BaseTransform
class Rotate(BaseTransform):
    def __init__(self, angle=0, mode='random'):
        super().__init__(mode)
        self.angle = angle

    def __call__(self, img, mask=None):
        h, w = img.shape[:2]

        if self.mode == 'random':
            angle = random.uniform(-self.angle, self.angle)
        else:
            angle = self.angle

        M = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1)

        img_rotated = cv2.warpAffine(img, M, (w, h), borderValue=255)

        if mask is not None:
            # Note: use nearest neighbor for masks to avoid interpolation artifacts
            mask_rotated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            return img_rotated, mask_rotated

        return img_rotated


class Translate(BaseTransform):
    def __init__(self, x=0, y=0, mode='random'):
        super().__init__(mode)
        self.x = x
        self.y = y

    def __call__(self, img, mask=None):
        h, w = img.shape[:2]

        if self.mode == 'random':
            tx = random.randint(-self.x, self.x)
            ty = random.randint(-self.y, self.y)
        else:
            tx = self.x
            ty = self.y

        M = np.float32([[1, 0, tx], [0, 1, ty]])

        img_translated = cv2.warpAffine(img, M, (w, h), borderValue=255)

        if mask is not None:
            mask_translated = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
            return img_translated, mask_translated

        return img_translated


class Scale(BaseTransform):
    def __init__(self, min_factor=0.9, max_factor=1.1, mode='random'):
        super().__init__(mode)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, img, mask=None):
        if img is None:
            raise ValueError("Input image is None")

        h, w = img.shape[:2]

        # Choose scaling factor
        if self.mode == 'random':
            scale = random.uniform(self.min_factor, self.max_factor)
        else:
            scale = (self.min_factor + self.max_factor) / 2

        # Resize image
        new_w = int(w * scale)
        new_h = int(h * scale)
        img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create canvas for image
        img_canvas = np.full((h, w), 255, dtype=np.uint8)

        # Compute placement
        offset_x = (w - new_w) // 2
        offset_y = (h - new_h) // 2

        src_x1 = max(0, -offset_x)
        src_y1 = max(0, -offset_y)
        dst_x1 = max(0, offset_x)
        dst_y1 = max(0, offset_y)

        src_x2 = min(new_w, w - dst_x1 + src_x1)
        src_y2 = min(new_h, h - dst_y1 + src_y1)

        img_canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
            img_resized[src_y1:src_y2, src_x1:src_x2]

        # Process mask if provided
        if mask is not None:
            mask_resized = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            mask_canvas = np.full((h, w), 0, dtype=mask.dtype)  # background label = 0
            mask_canvas[dst_y1:dst_y1 + (src_y2 - src_y1), dst_x1:dst_x1 + (src_x2 - src_x1)] = \
                mask_resized[src_y1:src_y2, src_x1:src_x2]
            return img_canvas, mask_canvas

        return img_canvas
