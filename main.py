import fiftyone.zoo as foz
import fiftyone as fo
from fiftyone.core.expressions import ViewField as F
import imgaug.augmenters as iaa
from PIL import Image
import numpy as np
import os
import json
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

# load the dataset
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    label_types=["detections"],
    classes=["person", "car"],
    max_samples=30000,
)

# filter the dataset to keep only "person" or "car" labels
filtered_view = dataset.filter_labels("ground_truth", F("label").is_in(["person", "car"]))

# further filter to include only samples that have at least one "person" or "car" detection
final_view = filtered_view.match(F("ground_truth.detections").length() > 0)

# Create a view that includes samples with both "person" and "car" labels
combined_view = final_view.match(
    (F("ground_truth.detections").filter(F("label") == "person").length() > 0) &
    (F("ground_truth.detections").filter(F("label") == "car").length() > 0)
)

# use 4000 samples
view = combined_view.limit(4000)

# augment both "person" and "car" samples (make sure boundary box is encoded for transformations)
# (x,y) top left encode to become -> (x,y) bottom right using imgaug
augmenter = iaa.Sequential([
    iaa.Affine(rotate=(-15, 15)),  # random rotation between -15 and 15 degrees
    iaa.Multiply((0.8, 1.2)),  # random brightness adjustment
    iaa.AdditiveGaussianNoise(scale=(0, 0.05*255)),  # random Gaussian noise
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
                augmenter = iaa.Sequential([
                    iaa.Affine(rotate=(-15, 15)),  # random rotation between -15 and 15 degrees
                    iaa.Multiply((0.8, 1.2)),  # random brightness adjustment
                    iaa.AdditiveGaussianNoise(scale=(0, 0.05 * 255)),  # random Gaussian noise
                ])

                # apply augmentations to image and bounding boxes
                # print(bbs_on_image)
                augmented = augmenter(image=img_array, bounding_boxes=bbs_on_image)
                aug_img = augmented[0]
                aug_bbs = augmented[1]

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
                aug_sample["tags"] = ["augmented"]

                # update bounding boxes in the augmented sample
                for det, aug_bbox in zip(aug_sample["ground_truth"].detections, aug_bbs.bounding_boxes):
                    # x1, y1 = aug_bbox.x1, aug_bbox.y1
                    # x2, y2 = aug_bbox.x2, aug_bbox.y2
                    # width = aug_bbox.x2 - aug_bbox.x1  # calculate width
                    # height = aug_bbox.y2 - aug_bbox.y1  # calculate height
                    # x2 = x1 + width  # add width to x1
                    # y2 = y1 + height  # add height to y1

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

    return augmented_samples

# augment samples with bounding boxes for "person" and "car"
augmented_dir = "augmented_samples"
os.makedirs(augmented_dir, exist_ok=True)

# augment both classes
augmented_person_samples = augment_samples_with_bboxes(view, "person", augmented_dir)
augmented_car_samples = augment_samples_with_bboxes(view, "car", augmented_dir)

# create a new dataset to hold the balanced and augmented samples
balanced_dataset = fo.Dataset()

# add original balanced samples
balanced_dataset.add_samples(list(view))

# add augmented samples
balanced_dataset.add_samples(augmented_person_samples)
balanced_dataset.add_samples(augmented_car_samples)

# launch app with the combined dataset (original and augmented samples)
session = fo.launch_app(balanced_dataset, port=5152)
session.wait()
