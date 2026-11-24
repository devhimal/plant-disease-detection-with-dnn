from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import tensorflow as tf

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def collect_image_paths(data_dir, crop_prefixes):
    data_dir = Path(data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {data_dir}\n"
            "Place the extracted PlantVillage class folders inside the project's data/PlantVillage folder."
        )

    class_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())
    selected = []
    for folder in class_dirs:
        name = folder.name.lower()
        if any(name.startswith(prefix.lower()) for prefix in crop_prefixes):
            selected.append(folder)

    if not selected:
        available = ", ".join(p.name for p in class_dirs[:30]) or "(no class folders found)"
        raise ValueError(
            "No Tomato, Potato, or Corn/Maize class folders were detected.\n"
            f"Folders found: {available}\n"
            "Check the dataset layout or edit CROP_PREFIXES in config.py."
        )

    paths, labels = [], []
    class_names = [p.name for p in selected]
    for label, folder in enumerate(selected):
        for file in folder.rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(str(file))
                labels.append(label)

    if not paths:
        raise ValueError("The selected class folders contain no supported image files.")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32), class_names

def make_datasets(data_dir, image_size, batch_size, validation_size, test_size, seed, crop_prefixes):
    paths, labels, class_names = collect_image_paths(data_dir, crop_prefixes)
    counts = np.bincount(labels)
    if np.min(counts) < 3:
        raise ValueError("Each selected class needs at least 3 images for stratified train/validation/test splitting.")

    train_paths, temp_paths, train_y, temp_y = train_test_split(
        paths, labels, test_size=validation_size + test_size,
        random_state=seed, stratify=labels
    )
    relative_test = test_size / (validation_size + test_size)
    val_paths, test_paths, val_y, test_y = train_test_split(
        temp_paths, temp_y, test_size=relative_test,
        random_state=seed, stratify=temp_y
    )

    def load_image(path, label):
        raw = tf.io.read_file(path)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    def build(paths_part, labels_part, training=False):
        ds = tf.data.Dataset.from_tensor_slices((paths_part, labels_part))
        if training:
            ds = ds.shuffle(min(len(paths_part), 10000), seed=seed, reshuffle_each_iteration=True)
        ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = build(train_paths, train_y, training=True)
    val_ds = build(val_paths, val_y)
    test_ds = build(test_paths, test_y)
    return train_ds, val_ds, test_ds, class_names, test_y
