import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
from config import IMAGE_SIZE, MODEL_DIR, RESULTS_DIR

def main():
    parser = argparse.ArgumentParser(description="Predict the plant leaf class from an image.")
    parser.add_argument("--image", required=True, help="Path to a leaf image")
    parser.add_argument("--model", default="cnn_relu_adam.keras", help="Model filename inside models/")
    args = parser.parse_args()

    model_path = MODEL_DIR / args.model
    model = tf.keras.models.load_model(model_path)
    class_names = (RESULTS_DIR / "class_names.txt").read_text(encoding="utf-8").splitlines()

    image = Image.open(args.image).convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    probabilities = model.predict(np.expand_dims(array, axis=0), verbose=0)[0]
    best = int(np.argmax(probabilities))
    print(f"Prediction: {class_names[best]}")
    print(f"Confidence: {probabilities[best] * 100:.2f}%")
    print("\nTop predictions:")
    for index in np.argsort(probabilities)[::-1][:5]:
        print(f"- {class_names[index]}: {probabilities[index] * 100:.2f}%")

if __name__ == "__main__":
    main()
