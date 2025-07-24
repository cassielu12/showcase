# augpipe 0.2 version

---
I design this python library sepcicifally for history document recognition tasks.
In the deadsea scoll recognition task, we use the augmentation algorithm in preprocess.py.

## Functions

- Geometric: Rotate, Scale, Translate
- Damage: BlackDropDamage, WhiteDropDamage, BlurDamage
- Distortion: Distortion
- YAML-driven pipelines via `load_pipeline_from_yaml`
---

## Structure

```
augpipe/
├── setup.py
├── requirements.txt
├── pipeline_loader.py      # YAML → ComposeTransform
├── wrappers.py             # AugupipeWrapper, ToTensorList (optional)
├── transforms/
│   ├── base.py
│   ├── geometric.py
│   ├── damage.py
│   ├── distortion.py
│   └── compose.py
├── configs/
│   ├── all_basic.yaml
│   └── one_of_all.yaml
└── utils.py                # Optional helpers
```

---

## Installation
after clone
```bash
cd augpipe
pip install -e .

```

---

## Testing

After installation:

try demo.py in example scripts 

---

## YAML Configuration Example

```yaml
mode: random
pipeline:
  - Rotate:
      angle: 8
  - Scale:
      min_factor: 0.85
      max_factor: 1.15
  - Translate:
      x: 3
      y: 2
  - Distortion:
      amp: 2.5
      sigma: 20.0
  - OneOf:
      - WhiteDropDamage:
          drop_length: 4
          drop_width: 4
      - BlackDropDamage:
          num_drops: 20
          min_len: 2
          max_len: 6
          thickness: 2
  - Sometimes:
      p: 0.5
      transform:
        BlurDamage:
          kernel: 7
          sigma: 2.0
```

----


## References 

- [Albumentations](https://albumentations.ai/)
- [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)
- [imgaug](https://imgaug.readthedocs.io/en/latest/)
- [OpenCV](https://opencv.org/)
- [Hydra / OmegaConf](https://hydra.cc/)
- [imagemorph.c](https://github.com/GrHound/imagemorph.c)

## Author:
https://github.com/pumpkinspicassie
