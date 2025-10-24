from export_to_yolo import export_to_yolo

export_to_yolo(
    dataset_name="person_car_final",
    label_field="ground_truth",
    export_dir="yolo_export",
    classes=["person", "car"],
    splits=["train", "val", "test"],
    ensure_aug_train=True,
    overwrite=True,
)