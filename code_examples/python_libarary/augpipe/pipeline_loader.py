import yaml
from .transforms.geometric import Rotate, Scale, Translate
from .transforms.damage import BlackDropDamage, WhiteDropDamage, BlurDamage
from .transforms.distortion import Distortion
from .transforms.compose import ComposeTransform, OneOfTransform, SometimesTransform
from .transforms import TRANSFORM_REGISTRY


def load_pipeline_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    mode = config.get("mode", "fixed")
    steps = config["pipeline"]
    transforms = []

    def parse_transform(step):
        if isinstance(step, dict):
            for name, params in step.items():
                if name in TRANSFORM_REGISTRY:
                    return TRANSFORM_REGISTRY[name](mode=mode, **params)
                elif name == "OneOf":
                    sub_transforms = [parse_transform(s) for s in params]
                    return OneOfTransform(sub_transforms, mode=mode)
                elif name == "Sometimes":
                    p = params["p"]
                    inner = parse_transform(params["transform"])
                    return SometimesTransform(inner, p=p, mode=mode)
                else:
                    raise ValueError(f"Unknown transform: {name}")
        else:
            raise ValueError("Each step must be a dict.")

    for step in steps:
        transforms.append(parse_transform(step))

    return ComposeTransform(transforms, mode=mode)
