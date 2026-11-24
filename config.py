from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "PlantVillage"
MODEL_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 2
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_SEED = 42

# The project focuses on tomato, potato, and corn/maize classes.
# Matching is case-insensitive and checks the beginning of each class folder name.
CROP_PREFIXES = ("tomato", "potato", "corn", "maize")

# Change to ("relu", "tanh") to run both activation experiments.
ACTIVATIONS = ("relu",)
OPTIMIZERS = ("adam", "rmsprop")
RUN_ANN = True
RUN_CNN = True
