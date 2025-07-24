#white drop part and blur part are modified from https://towardsdatascience.com/effective-data-augmentation-for-ocr-8013080aa9fa/

import random
import cv2
import numpy as np
import albumentations as A
from .base import BaseTransform

class BlackDropDamage(BaseTransform):
    def __init__(self, num_drops=5, min_len=5, max_len=15, thickness=2, mode='random'):
        super().__init__(mode)
        self.num_drops = num_drops
        self.min_len = min_len
        self.max_len = max_len
        self.thickness = thickness

    def __call__(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if self.mode == 'random':
            num_drops = random.randint(1, self.num_drops)
            thickness = random.randint(0, self.thickness)
        else:
            num_drops = self.num_drops
            thickness = self.thickness
        for _ in range(num_drops):
            x = random.randint(0, w - 1)
            y = random.randint(0, h - 1)

            if random.random() < 0.5:
                # draw a black line (rain streak)
                length = random.randint(self.min_len, self.max_len)
                angle = random.uniform(-10, 10)
                thickness = random.randint(1, self.thickness)
                x_end = int(x + length * np.sin(np.deg2rad(angle)))
                y_end = int(y + length * np.cos(np.deg2rad(angle)))
                cv2.line(mask, (x, y), (x_end, y_end), color=255, thickness=thickness)
            else:
                # draw a black dot (ink drop)
                radius = random.randint(1, 4)
                cv2.circle(mask, (x, y), radius=radius, color=255, thickness=-1)
        mask = cv2.GaussianBlur(mask, (3, 3), sigmaX=1.0)
        rain_layer = np.stack([mask] * 3, axis=-1)
        img = np.where(rain_layer == 255, 0, img)

        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

class WhiteDropDamage(BaseTransform):
    def __init__(self,
                 drop_length=5,
                 drop_width=3,
                 p=1.0,
                 mode='random'):
        super().__init__(mode)
        self.drop_length = drop_length
        self.drop_width = drop_width
        self.p = p

    def __call__(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.mode == 'random':
            drop_length = random.randint(1, self.drop_length)
            drop_width = random.randint(1, self.drop_width)
            p=self.p
        else:
            drop_length = self.drop_length
            drop_width = self.drop_width
            p = self.p
        transform = A.RandomRain(
            brightness_coefficient=1.0,
            drop_length=drop_length,
            drop_width=drop_width,
            drop_color=(255, 255, 255),
            blur_value=1,
            rain_type='drizzle',
            p=p
        )

        augmented = transform(image=img)['image']
        return cv2.cvtColor(augmented, cv2.COLOR_BGR2GRAY)


class BlurDamage(BaseTransform):
    def __init__(self, kernel=5, sigma=1.5, mode='random'):
        super().__init__(mode)
        self.kernel = kernel
        self.sigma = sigma
        assert isinstance(kernel, int), "kernel must be an integer"

    def __call__(self, img):
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        if self.mode == 'random':
            # kernel ∈ [3, kernel_max] 且必须为奇数
            ksize = random.choice([k for k in range(3, self.kernel+ 1) if k % 2 == 1])
            sigma = random.uniform(0.1, self.sigma)
        else:
            ksize = self.kernel
            sigma = self.sigma

        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigmaX=sigma)
        return cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

