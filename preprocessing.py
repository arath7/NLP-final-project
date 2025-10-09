import os
import json
import csv
import random
import pandas as pd

TRAIN_META = r"nuimages-v1.0-all-metadata\v1.0-train"
VAL_META   = r"nuimages-v1.0-all-metadata\v1.0-val"

IMAGES_ROOT = r"C:\Users\anany\OneDrive\Desktop\nuimages-v1.0-all-samples\samples"

OUTPUT_DIR = r"nuimages_csv_splits"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def build_lookup(lst, key="token"):
    return {x[key]: x for x in lst}



samples = json.load(open(os.path.join(TRAIN_META, "sample.json"))) + json.load(open(os.path.join(VAL_META, "sample.json")))
sample_data = json.load(open(os.path.join(TRAIN_META, "sample_data.json"))) + json.load(open(os.path.join(VAL_META, "sample_data.json")))
surface_anns = json.load(open(os.path.join(TRAIN_META, "surface_ann.json"))) + json.load(open(os.path.join(VAL_META, "surface_ann.json")))
object_anns = json.load(open(os.path.join(TRAIN_META, "object_ann.json"))) + json.load(open(os.path.join(VAL_META, "object_ann.json")))
ego_poses = json.load(open(os.path.join(TRAIN_META, "ego_pose.json"))) + json.load(open(os.path.join(VAL_META, "ego_pose.json")))
categories = json.load(open(os.path.join(TRAIN_META, "category.json")))
calibrated_sensors = json.load(open(os.path.join(TRAIN_META, "calibrated_sensor.json"))) + json.load(open(os.path.join(VAL_META, "calibrated_sensor.json")))

sample_lookup = build_lookup(samples)
sample_data_lookup = build_lookup(sample_data)
category_lookup = build_lookup(categories)
ego_lookup = build_lookup(ego_poses)
sensor_lookup = build_lookup(calibrated_sensors)

sd_to_surface = {}
for ann in surface_anns:
    sd_to_surface.setdefault(ann["sample_data_token"], []).append(ann)

sd_to_objects = {}
for ann in object_anns:
    sd_to_objects.setdefault(ann["sample_data_token"], []).append(ann)


all_sd_tokens = list(set(list(sd_to_surface.keys()) + list(sd_to_objects.keys())))
random.shuffle(all_sd_tokens)



n_total = len(all_sd_tokens)
n_train = int(n_total * TRAIN_RATIO)
n_val = int(n_total * VAL_RATIO)

splits = {
    "train": all_sd_tokens[:n_train],
    "val":   all_sd_tokens[n_train:n_train+n_val],
    "test":  all_sd_tokens[n_train+n_val:]
}

print(f"Split sizes → Train: {len(splits['train'])}, Val: {len(splits['val'])}, Test: {len(splits['test'])}")



def write_csv(split_name, sd_tokens):
    csv_path = os.path.join(OUTPUT_DIR, f"{split_name}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "image_path",
            "sample_token",
            "camera",
            "width",
            "height",
            "timestamp",
            "location_token",
            "ego_translation",
            "ego_rotation",
            "surface_categories",
            "surface_masks",
            "object_categories",
            "bboxes",
            "object_attributes"
        ])
        
        for sd_token in sd_tokens:
            sd = sample_data_lookup[sd_token]
            sample_token = sd["sample_token"]
            image_path = os.path.join("nuimages-v1.0-all-samples", "samples", sd["filename"].replace("\\", "/"))
            
            camera = sd.get("channel", "")
            width = sd.get("width", "")
            height = sd.get("height", "")
            timestamp = sd.get("timestamp", "")
            
            sample_info = sample_lookup[sample_token]
            location_token = sample_info.get("location_token", "")
            
            ego = ego_lookup.get(sd["ego_pose_token"], {})
            ego_translation = json.dumps(ego.get("translation", []))
            ego_rotation = json.dumps(ego.get("rotation", []))
            
            surface_list = sd_to_surface.get(sd_token, [])
            surface_categories = json.dumps([category_lookup[a["category_token"]]["name"] for a in surface_list])
            surface_masks = json.dumps([a.get("mask", "") for a in surface_list])
            
            obj_list = sd_to_objects.get(sd_token, [])
            object_categories = json.dumps([category_lookup[o["category_token"]]["name"] for o in obj_list])
            bboxes = json.dumps([o["bbox"] for o in obj_list])
            object_attributes = json.dumps([o.get("attribute", "") for o in obj_list])
            
            writer.writerow([
                image_path,
                sample_token,
                camera,
                width,
                height,
                timestamp,
                location_token,
                ego_translation,
                ego_rotation,
                surface_categories,
                surface_masks,
                object_categories,
                bboxes,
                object_attributes
            ])
    print(f"Wrote {csv_path}")


for split_name, tokens in splits.items():
    write_csv(split_name, tokens)
