# Plant Disease Detection with Deep Neural Networks

This project compares a Convolutional Neural Network (CNN) and a fully connected Artificial Neural Network (ANN) for classifying PlantVillage leaf images. It focuses on tomato, potato, and corn/maize class folders and compares Adam with RMSprop.

## 1. Project contents

- `train.py` — trains the configured CNN/ANN experiments and evaluates each model.
- `predict.py` — predicts the class of one leaf image using a saved model.
- `config.py` — dataset path, image size, epochs, activation and optimizer settings.
- `src/data.py` — scans folders, splits the data into training/validation/test sets and builds TensorFlow datasets.
- `src/models.py` — CNN and ANN architecture definitions.
- `src/evaluate.py` — training curves, confusion matrices and classification reports.
- `results/` — generated plots, reports and CSV summary.
- `models/` — saved Keras models.

## 2. Dataset setup

Download the PlantVillage dataset from a trusted source (for example, the PlantVillage dataset on Kaggle), then extract it. Place the class folders directly under:

```text
plant_disease_detection_project/
└── data/
    └── PlantVillage/
        ├── Tomato___Early_blight/
        ├── Tomato___healthy/
        ├── Potato___Late_blight/
        ├── Corn_(maize)___Common_rust/
        └── ...
```

The code selects folders whose names begin with `Tomato`, `Potato`, `Corn`, or `Maize` (case-insensitive). If your extracted dataset has an extra parent folder, move the class folders into `data/PlantVillage`, or change `DATA_DIR` in `config.py`. If you only want two crops, edit `CROP_PREFIXES`.

The selected folders should contain image files. Do not put the entire dataset ZIP file in the folder without extracting it.

## 3. Installation

Use Python 3.10 or 3.11 for a straightforward TensorFlow setup.

### Windows PowerShell

```powershell
cd plant_disease_detection_project
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks environment activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\Activate.ps1
```

### macOS / Linux

```bash
cd plant_disease_detection_project
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Run training

```bash
python train.py
```

The default settings run CNN and ANN with ReLU and compare Adam and RMSprop. Training may take a long time on a CPU. A compatible GPU can speed it up, but TensorFlow GPU setup depends on the operating system and hardware.

To test Tanh too, change this line in `config.py`:

```python
ACTIVATIONS = ("relu", "tanh")
```

To run only the CNN or only the ANN, set `RUN_ANN` or `RUN_CNN` to `False`. For a quick software test, temporarily set `EPOCHS = 2`; use a larger epoch count for the final experiment.

## 5. Outputs

After training, check:

- `results/experiment_summary.csv` — accuracy, weighted precision, weighted recall and weighted F1 for each experiment.
- `results/*_training_curves.png` — training and validation loss/accuracy.
- `results/*_confusion_matrix.png` — confusion matrix for each experiment.
- `results/*_classification_report.txt` — precision, recall and F1 by class.
- `models/*.keras` — trained models.

Use the actual generated outputs in your assignment report. Do not report example or estimated scores as experimental results.

## 6. Predict a new image

After training, run:

```bash
python predict.py --image "path/to/leaf.jpg" --model cnn_relu_adam.keras
```

The model predicts a PlantVillage class label. This is an educational image classifier, not a substitute for professional agricultural diagnosis.

## 7. Methodology

- Images are resized to 128 × 128 pixels and normalized to the 0–1 range.
- Data is divided into training, validation and test partitions using stratified splitting.
- The CNN uses convolution, pooling, batch normalization, dropout and global average pooling.
- The ANN uses flattened image pixels and dense layers.
- The output layer uses softmax; the hidden layers use the selected ReLU or Tanh activation.
- Sparse categorical cross-entropy is the loss function.
- Adam and RMSprop are compared as gradient-based optimizers.
- Early stopping and learning-rate reduction are used to help control overfitting and training stability.

## 8. Important notes

1. Run the project from the project root directory.
2. The first run can use substantial RAM and disk space. If memory is limited, reduce `BATCH_SIZE` to 16 and/or `IMAGE_SIZE` to `(96, 96)`.
3. ANN training on flattened images can be slower and use more memory than CNN training.
4. Dataset class names are used as labels. Keep the generated `results/class_names.txt` together with the model.
5. Results depend on the dataset version, class distribution, hardware, and training configuration.
# plant-disease-detection-with-dnn
