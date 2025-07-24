import torch
import os
import cv2
import requests
import uuid
import json
import numpy as np
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.structures import Boxes, Instances

# Make sure this list matches your 8 trained classes + "undefined" fallback
MetadataCatalog.get("doclayout_train").set(thing_classes=[
    "heading", "paragraph", "request", "decision",
    "marginalia", "attendance_list", "catch_word", "date", "undefined"
])

def reclassify(image_crop, original_label):
    """
    Sends one cropped patch to your re‐classification server and returns
    (new_label_id, new_label_name). On error, returns ("undefined" id, "undefined").
    """
    thing_classes = MetadataCatalog.get("doclayout_train").thing_classes
    label2id = {name: idx for idx, name in enumerate(thing_classes)}
    temp_path = f"/tmp/{uuid.uuid4().hex}.png"
    cv2.imwrite(temp_path, image_crop)
    try:
        with open(temp_path, "rb") as f:
            response = requests.post(
                "http://localhost:5001/reclassify",
                files={"file": f}
            )
            result = response.json()
    except Exception as e:
        print("[ERROR] Failed to reach reclassify service:", e)
        return label2id["undefined"], "undefined"

    text = result.get("text", "error")
    label = result.get("class", "error")
    if text == "error" or label == "error" or label not in label2id:
        return label2id["undefined"], "undefined"
    else:
        return label2id[label], label

def setup_cfg(model_path="models/model_final.pth"):
    """
    Standard Detectron2 config. 
    Set NUM_CLASSES=8 (since you trained on exactly 8 classes), and set the weights.
    """
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = model_path
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg

def run_inference(image_path, output_dir, model_path):
    """
    For each image (or every image in a folder), this will:
      1) Run Detectron2 inference
      2) Re‐classify low‐confidence boxes if needed
      3) Collect all “final” instances (skipping label="undefined")
      4) Write ONE JSON file in exactly this format:
         {
           "image_name": "<basename + extension>",
           "objects": [
             { "label": "<label_name>", "bbox": [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] },
             …
           ]
         }
      5) Write ONE per‐class mask PNG named "<basename>.png" into a separate folder
    """

    cfg = setup_cfg(model_path)
    predictor = DefaultPredictor(cfg)
    thing_classes = MetadataCatalog.get("doclayout_train").thing_classes

    # 1) gather all image files
    if os.path.isdir(image_path):
        img_files = [
            os.path.join(root, f)
            for root, _, files in os.walk(image_path)
            for f in files
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
    else:
        img_files = [image_path]  # single image

    # Prepare output subdirectories
    output_dir_json = os.path.join(output_dir, "json")
    output_dir_mask = os.path.join(output_dir, "segmentation_masks")
    output_dir_images = os.path.join(output_dir, "images")
    os.makedirs(output_dir_json, exist_ok=True)
    os.makedirs(output_dir_mask, exist_ok=True)
    os.makedirs(output_dir_images, exist_ok=True)

    for img_file in img_files:
        print(f"Running inference on: {img_file}")
        image = cv2.imread(img_file)
        if image is None:
            print(f"[WARN] Could not read {img_file}, skipping.")
            continue

        # Save a copy of the original image (optional)
        cv2.imwrite(os.path.join(output_dir_images, os.path.basename(img_file)), image)

        outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        H, W = image.shape[:2]
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else None

        # Build final JSON structure
        base_name = os.path.splitext(os.path.basename(img_file))[0]
        json_dict = {
            "image_name": os.path.basename(img_file),
            "objects": []
        }

        # We'll also collect masks for producing a single per‐pixel mask
        mask_list = []

        for i in range(len(instances)):
            # Raw bounding box = [x1, y1, x2, y2] (floats)
            box = instances.pred_boxes[i].tensor.numpy()[0].tolist()
            score = float(instances.scores[i])
            original_label_id = int(instances.pred_classes[i])
            label_id = original_label_id
            label_name = thing_classes[label_id]

            # Re‐classify if low confidence and ambiguous class
            if score < 0.9 and label_name in [
                "attendance_list", "decision", "request", "paragraph", "heading"
            ]:
                x1, y1, x2, y2 = map(int, box)
                crop_img = image[y1:y2, x1:x2]
                new_id, new_name = reclassify(crop_img, label_name)
                label_id = new_id
                label_name = new_name

            # Skip “undefined”
            if label_name == "undefined":
                continue

            # Convert box floats → ints, and build polygon as [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            polygon = [
                [x1, y1],
                [x2, y1],
                [x2, y2],
                [x1, y2]
            ]

            # Append to JSON “objects” list
            json_dict["objects"].append({
                "label": label_name,
                "bbox": polygon
            })

            # Also stash the mask for building the combined mask‐PNG
            if masks is not None:
                boolean_mask = masks[i].astype(bool)
                mask_list.append((boolean_mask, label_id))

        # 2) Write the JSON file in the correct format:
        json_path = os.path.join(output_dir_json, f"{base_name}.json")
        with open(json_path, "w") as f:
            json.dump(json_dict, f, indent=2)

        # 3) Build and save a single per‐pixel mask PNG
        if mask_list:
            combined_mask = np.zeros((H, W), dtype=np.uint8)
            for boolean_mask, lid in mask_list:
                # pixel value = lid+1 (so background stays 0)
                combined_mask[boolean_mask] = lid + 1

            mask_path = os.path.join(output_dir_mask, f"{base_name}.png")
            # Write with zero compression to preserve raw values (0,1,2,…)
            cv2.imwrite(mask_path, combined_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            # No instances ⇒ write an all‐zero mask
            empty_mask = np.zeros((H, W), dtype=np.uint8)
            mask_path = os.path.join(output_dir_mask, f"{base_name}.png")
            cv2.imwrite(mask_path, empty_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        print(f"  → JSON saved to: {json_path}")
        print(f"  → Mask PNG saved to: {mask_path}")
