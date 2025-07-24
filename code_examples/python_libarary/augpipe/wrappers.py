
# augupipe/wrappers.py
from PIL import Image
import numpy as np

class AugupipeWrapper:
    def __init__(self, pipeline):
        self.pipeline = pipeline

    def __call__(self, img: Image.Image) -> Image.Image:
        img_np = np.array(img)
        aug_np = self.pipeline(img_np)
        return Image.fromarray(aug_np) 