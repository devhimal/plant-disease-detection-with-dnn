import json
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from config import (
    DATA_DIR, MODEL_DIR, RESULTS_DIR, IMAGE_SIZE, BATCH_SIZE, EPOCHS,
    VALIDATION_SIZE, TEST_SIZE, RANDOM_SEED, CROP_PREFIXES,
    ACTIVATIONS, OPTIMIZERS, RUN_ANN, RUN_CNN
)
from src.data import make_datasets
from src.models import build_cnn, build_ann, get_optimizer
from src.evaluate import save_training_curves, evaluate_model

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    tf.random.set_seed(RANDOM_SEED)

    train_ds, val_ds, test_ds, class_names, y_test = make_datasets(
        DATA_DIR, IMAGE_SIZE, BATCH_SIZE, VALIDATION_SIZE, TEST_SIZE,
        RANDOM_SEED, CROP_PREFIXES
    )
    (RESULTS_DIR / "class_names.txt").write_text("\n".join(class_names), encoding="utf-8")
    print(f"Detected {len(class_names)} classes and prepared the dataset.")
    print("Classes:", class_names)

    results = []
    experiments = []
    for model_type, enabled in (("CNN", RUN_CNN), ("ANN", RUN_ANN)):
        if not enabled:
            continue
        for activation in ACTIVATIONS:
            for optimizer_name in OPTIMIZERS:
                experiments.append((model_type, activation, optimizer_name))

    for model_type, activation, optimizer_name in experiments:
        run_name = f"{model_type.lower()}_{activation}_{optimizer_name}"
        print(f"\n{'='*70}\nStarting experiment: {run_name}\n{'='*70}")

        input_shape = (*IMAGE_SIZE, 3)
        if model_type == "CNN":
            model = build_cnn(input_shape, len(class_names), activation)
        else:
            model = build_ann(input_shape, len(class_names), activation)

        model.compile(
            optimizer=get_optimizer(optimizer_name),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=4, restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6
            ),
        ]
        history = model.fit(
            train_ds, validation_data=val_ds, epochs=EPOCHS,
            callbacks=callbacks, verbose=1
        )
        save_training_curves(history, RESULTS_DIR, run_name)
        model.save(MODEL_DIR / f"{run_name}.keras")
        metrics = evaluate_model(model, test_ds, y_test, class_names, RESULTS_DIR, run_name)
        metrics.update({"model": model_type, "activation": activation, "optimizer": optimizer_name})
        results.append(metrics)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_DIR / "experiment_summary.csv", index=False)
    print("\nExperiment summary:")
    print(results_df.to_string(index=False))
    print(f"\nSaved models to: {MODEL_DIR}")
    print(f"Saved reports and plots to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
