import fiftyone.zoo as foz
import fiftyone as fo
from fiftyone.core.expressions import ViewField as F
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import os
import json
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import random
import gc
from export_to_yolo import export_to_yolo
import fiftyone.types as types

SEED = 42
VAL_COUNT = 500
TEST_COUNT = 500
BOTH_TOTAL_TARGET = 369

# load the train dataset
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car"],
    max_samples=30000,
)

# filter the dataset to keep only "person" or "car" labels
filtered_view = train_dataset.filter_labels("ground_truth", F("label").is_in(["person", "car"]))

# further filter to include only samples that have at least one "person" or "car" detection
final_view = filtered_view.match(F("ground_truth.detections").length() > 0)

# Create a view that includes samples with both "person" and "car" labels
combined_view = final_view.match(
    (F("ground_truth.detections").filter(F("label") == "person").length() > 0) &
    (F("ground_truth.detections").filter(F("label") == "car").length() > 0)
)

# use 4000 samples
view = combined_view.limit(4000)

val_ds = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["detections"],
    classes=["person", "car"],
)
val_filtered = val_ds.filter_labels("ground_truth", F("label").is_in(["person", "car"]))

both_view = val_filtered.match(
    (F("ground_truth.detections").filter(F("label") == "person").length() > 0) &
    (F("ground_truth.detections").filter(F("label") == "car").length() > 0)
)
person_only_view = val_filtered.match(
    (F("ground_truth.detections").filter(F("label") == "person").length() > 0) &
    (F("ground_truth.detections").filter(F("label") == "car").length() == 0)
)
car_only_view = val_filtered.match(
    (F("ground_truth.detections").filter(F("label") == "car").length() > 0) &
    (F("ground_truth.detections").filter(F("label") == "person").length() == 0)
)

rng = random.Random(SEED)
both_ids = list(both_view.values("id"))
person_only_ids = list(person_only_view.values("id"))
car_only_ids = list(car_only_view.values("id"))
single_ids = person_only_ids + car_only_ids

rng.shuffle(both_ids)
rng.shuffle(single_ids)

need_val = VAL_COUNT
need_test = TEST_COUNT
need_total = need_val + need_test

desired_both_total = min(BOTH_TOTAL_TARGET, len(both_ids), need_total)
val_both_target = min(desired_both_total // 2 + desired_both_total % 2, need_val)
test_both_target = desired_both_total - val_both_target

val_both_ids = both_ids[:val_both_target]
test_both_ids = both_ids[val_both_target:val_both_target + test_both_target]

remaining_val = need_val - len(val_both_ids)
remaining_test = need_test - len(test_both_ids)

if len(single_ids) < (remaining_val + remaining_test):
    raise ValueError(f"Not enough single-class images to fill: need {remaining_val + remaining_test}, have {len(single_ids)}")

val_single_ids = single_ids[:remaining_val]
test_single_ids = single_ids[remaining_val:remaining_val + remaining_test]

val_ids = val_both_ids + val_single_ids
test_ids = test_both_ids + test_single_ids

# safety uniqueness
val_set = set(val_ids)
test_ids = [i for i in test_ids if i not in val_set][:need_test]

assert len(val_ids) == need_val
assert len(test_ids) == need_test
assert not (set(val_ids) & set(test_ids))

val_view = val_filtered.select(val_ids)
test_view = val_filtered.select(test_ids)

print(f"[info] val: total={len(val_view)} both={len(val_both_ids)} single={len(val_single_ids)}")
print(f"[info] test: total={len(test_view)} both={len(test_both_ids)} single={len(test_single_ids)}")

# tag splits persistently using views (previous calls caused TypeError)
view.tag_samples("train")
val_view.tag_samples("val")
test_view.tag_samples("test")

# augment both "person" and "car" samples (make sure boundary box is encoded for transformations)
# (x,y) top left encode to become -> (x,y) bottom right using imgaug
AUGMENTER = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),
    iaa.Multiply((0.8, 1.2)),
    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),
])
augmented_dir = "augmented_samples"
os.makedirs(augmented_dir, exist_ok=True)

def normalize_image_size(image_path, output_path, max_size=640):
    """
    Normalizes the size of an image to ensure its width and height are below max_size.

    Args:
        image_path (str): Path to the input image.
        output_path (str): Path to save the normalized image.
        max_size (int): Maximum size for width and height (default is 640).

    Returns:
        None
    """
    with Image.open(image_path) as img:
        # get original dimensions
        width, height = img.size

        # calculate scaling factor to maintain aspect ratio
        scaling_factor = min(max_size / width, max_size / height)

        # resize image if necessary
        if scaling_factor < 1:
            new_width = int(width * scaling_factor)
            new_height = int(height * scaling_factor)
            img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # save the normalized image
        img.save(output_path)

def augment_samples_with_bboxes(view, label, augmented_dir, num_samples=200, augmentations_per_sample=5, online=False):
    """
        Augments samples in a view with bounding boxes and saves the augmented images/bbs
        Args:
            view (fiftyone.core.view.DatasetView): The view containing samples to augment
            label (str): The label type to augment (e.g., "person" or "car")
            augmented_dir (str): Directory to save augmented images and bounding boxes
            num_samples (int): Number of samples to augment
            augmentations_per_sample (int): Number of augmentations per sample
            online (bool): If True, does not save images to disk, only returns augmented samples
    """
    augmented_samples = []

    for sample in view.limit(num_samples):
        # load and normalize image
        normalized_filepath = os.path.join(augmented_dir, f"{sample.id}_normalized.jpg")
        normalize_image_size(sample.filepath, normalized_filepath, max_size=640)
        image = Image.open(normalized_filepath)
        img_array = np.array(image)

        # extract bounding boxes and labels (make sure encoding specifically for ground truth topleft/bottomright OR center)
        detections = sample["ground_truth"].detections

        # convert bounding boxes from normalized (x, y, w, h) to (x1, y1, x2, y2) with absolute pixel values and w/h from image size
        labels = [
            [
                det.label,
                det.bounding_box[0] * img_array.shape[1],  # x1 (x * img width)
                det.bounding_box[1] * img_array.shape[0],  # y1 (y * img height)
                (det.bounding_box[0] + det.bounding_box[2]) * img_array.shape[1],  # x2 (x + w) * img width
                (det.bounding_box[1] + det.bounding_box[3]) * img_array.shape[0],  # y2 (y + h) * img height
            ]
            for det in detections
        ]

        # convert bounding boxes to imgaug format (top-left and bottom-right)
        bbs = [
            BoundingBox(label=label[0], x1=label[1], y1=label[2], x2=label[3], y2=label[4])
            for label in labels
        ]
        bbs_on_image = BoundingBoxesOnImage(bbs, shape=img_array.shape)

        # apply augmentations
        for i in range(augmentations_per_sample):
            try:
                aug_img, aug_bbs = AUGMENTER(image=img_array, bounding_boxes=bbs_on_image)
                # save augmented image offline if required
                aug_filepath = os.path.join(augmented_dir, f"{sample.id}_{label}_aug_{i}.jpg")
                if not online:
                    Image.fromarray(aug_img).save(aug_filepath)

                # save bbs in original format
                bboxes_data = [
                    {
                        "label": str(aug_bbox.label),
                        "x1": float(aug_bbox.x1),
                        "y1": float(aug_bbox.y1),
                        "x2": float(aug_bbox.x2),
                        "y2": float(aug_bbox.y2),
                    }
                    for aug_bbox in aug_bbs.bounding_boxes
                ]
                bboxes_filepath = os.path.join(augmented_dir, f"{sample.id}_{label}_aug_{i}_bboxes.json")
                with open(bboxes_filepath, "w") as f:
                    json.dump(bboxes_data, f, indent=4)

                # create a new sample for the augmented image
                aug_sample = fo.Sample(filepath=aug_filepath if not online else None)
                aug_sample["ground_truth"] = sample["ground_truth"].copy()
                aug_sample.tags = list(dict.fromkeys(sample.tags + ["augmented"]))  # inherit + add augmented

                # update bounding boxes in the augmented sample
                for det, aug_bbox in zip(aug_sample["ground_truth"].detections, aug_bbs.bounding_boxes):
                    # convert (x1, y1, x2, y2) to normalized coordinates (x, y, w, h) for FiftyOne
                    x1, y1 = aug_bbox.x1, aug_bbox.y1
                    x2, y2 = aug_bbox.x2, aug_bbox.y2
                    img_w, img_h = aug_img.shape[1], aug_img.shape[0]
                    # ensure top-left coordinates are >= 0 and normalized
                    x = max(0, x1 / img_w)
                    y = max(0, y1 / img_h)
                    # ensure width and height are <= 1 and normalized
                    w = min(1, (x2 - x1) / img_w)
                    h = min(1, (y2 - y1) / img_h)
                    det.bounding_box = [x, y, w, h]
                    det.label = aug_bbox.label  # ensure label is augmented

                augmented_samples.append(aug_sample)

            except Exception as e:
                print(f"Error augmenting sample {sample.id}: {e}")

        del img_array
        gc.collect()

    return augmented_samples

# augment samples with bounding boxes for "person" and "car"
augmented_dir = "augmented_samples"
os.makedirs(augmented_dir, exist_ok=True)

# augment both classes
augmented_person_samples = augment_samples_with_bboxes(view, "person", augmented_dir)
augmented_car_samples = augment_samples_with_bboxes(view, "car", augmented_dir)

balanced_dataset = fo.Dataset(name="person_car_final")
balanced_dataset.add_samples(list(view))
balanced_dataset.add_samples(list(val_view))
balanced_dataset.add_samples(list(test_view))
balanced_dataset.add_samples(augmented_person_samples)
balanced_dataset.add_samples(augmented_car_samples)

print("[info] counts after building balanced dataset:")
print("  train:", balanced_dataset.match_tags("train").count())
print("  val:", balanced_dataset.match_tags("val").count())
print("  test:", balanced_dataset.match_tags("test").count())
print("  augmented:", balanced_dataset.match_tags("augmented").count())

# export all distinct tags (train/val/test/augmented) to YOLO
all_tags = balanced_dataset.distinct("tags")
print(f"[info] exporting tags -> {all_tags}")
export_to_yolo(
    dataset_name="person_car_final",
    label_field="ground_truth",
    export_dir="yolo_export",
    classes=["person", "car"],
    splits=all_tags,
    overwrite=True,
)
print("[info] export complete to yolo_export/")
