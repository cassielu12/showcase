
# Package entry point. This can be extended to expose selected augmentations
# or configuration settings if needed.
from .geometric import Rotate, Translate, Scale
from .damage import WhiteDropDamage, BlackDropDamage, BlurDamage
from .distortion import Distortion
from .compose import ComposeTransform, OneOfTransform, SometimesTransform

__all__ = [
    'Rotate', 'Translate', 'Scale',
    'WhiteDropDamage', 'BlackDropDamage', 'BlurDamage',
    'Distortion',
    'ComposeTransform', 'OneOfTransform', 'SometimesTransform'
]


TRANSFORM_REGISTRY = {
    "Rotate": Rotate,
    "Scale": Scale,
    "Translate": Translate,
    "Distortion": Distortion,
    "BlackDropDamage": BlackDropDamage,
    "WhiteDropDamage": WhiteDropDamage,
    "BlurDamage": BlurDamage,
}