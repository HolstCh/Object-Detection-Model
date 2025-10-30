import os
import shutil
import fiftyone as fo
import fiftyone.types as types

def export_to_yolo(
    dataset_name: str,
    label_field: str,
    export_dir: str,
    classes: list,
    splits: list,
    ensure_aug_train: bool = True,
    overwrite: bool = False,
):
    ds = fo.load_dataset(dataset_name)

    if ensure_aug_train:
        changed = False
        for s in ds.match_tags("augmented"):
            if "train" not in s.tags:
                s.tags.append("train")
                s.save()
                changed = True
        if changed:
            print("Info: added train tag to augmented samples")

    if overwrite and os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.makedirs(export_dir, exist_ok=True)

    ds.compute_metadata()

    classes = [c for c in classes if c]
    splits = [sp for sp in splits if sp]

    first = True
    for split in splits:
        view = ds.match_tags(split)
        count = len(view)
        if count == 0:
            print(f"Warn: split {split} empty, skipped")
            continue
        print(f"Exporting {split} ({count} samples)")
        view.export(
            export_dir=export_dir,
            dataset_type=types.YOLOv5Dataset,
            label_field=label_field,
            split=split,
            classes=classes,
            overwrite=first,
        )
        first = False

    yaml_path = os.path.join(export_dir, "data.yaml")
    if not os.path.exists(yaml_path):
        with open(yaml_path, "w") as f:
            f.write("names:\n")
            for i, name in enumerate(classes):
                f.write(f"  {i}: {name}\n")
        print("Created data.yaml")

    print(f"Done: YOLO export at {export_dir}")