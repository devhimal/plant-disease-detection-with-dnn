import tensorflow as tf
from tensorflow.keras import layers, models

def build_cnn(input_shape, num_classes, activation="relu"):
    inputs = layers.Input(shape=input_shape)
    x = layers.RandomFlip("horizontal")(inputs)
    x = layers.RandomRotation(0.08)(x)

    for filters in (32, 64, 128):
        x = layers.Conv2D(filters, 3, padding="same", activation=activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same", activation=activation)(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.20)(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation=activation)(x)
    x = layers.Dropout(0.40)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=f"CNN_{activation}")

def build_ann(input_shape, num_classes, activation="relu"):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(512, activation=activation)(x)
    x = layers.Dropout(0.35)(x)
    x = layers.Dense(256, activation=activation)(x)
    x = layers.Dropout(0.30)(x)
    x = layers.Dense(128, activation=activation)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs, outputs, name=f"ANN_{activation}")

def get_optimizer(name):
    name = name.lower()
    if name == "adam":
        return tf.keras.optimizers.Adam(learning_rate=0.001)
    if name == "rmsprop":
        return tf.keras.optimizers.RMSprop(learning_rate=0.001)
    raise ValueError(f"Unsupported optimizer: {name}")
