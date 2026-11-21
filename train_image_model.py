import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "image_dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(exist_ok=True)

# Training settings
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 32
SEED = 42

print("Loading disaster image dataset...")

train_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="training",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

validation_data = tf.keras.utils.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=0.20,
    subset="validation",
    seed=SEED,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_data.class_names

print("Classes:", class_names)

# Save class names for Flask prediction later
with open(MODEL_DIR / "image_classes.json", "w") as file:
    json.dump(class_names, file)

# Improve data loading
AUTOTUNE = tf.data.AUTOTUNE
train_data = train_data.prefetch(buffer_size=AUTOTUNE)
validation_data = validation_data.prefetch(buffer_size=AUTOTUNE)

# Data augmentation
augmentation = models.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.10)
])

# Pretrained image model
base_model = MobileNetV2(
    input_shape=(160, 160, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

inputs = layers.Input(shape=(160, 160, 3))
x = augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Save best model
checkpoint = ModelCheckpoint(
    MODEL_DIR / "disaster_image_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    verbose=1
)

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
)

print("\nStarting image model training...")

history = model.fit(
    train_data,
    validation_data=validation_data,
    epochs=5,
    callbacks=[checkpoint, early_stop]
)

# Save final model
model.save(MODEL_DIR / "disaster_image_model.keras")

print("\nTraining completed!")
print("Model saved inside the models folder.")
print("Image classes:", class_names)