# preprocess.py
# This script loads, preprocesses, and verifies your dataset
# It does NOT save images — TensorFlow handles augmentation during training

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── Configuration ────────────────────────────────────────────────────────────
DATASET_PATH = "dataset"
IMG_SIZE     = (224, 224)   # MobileNetV2 standard input size
BATCH_SIZE   = 16           # How many images to process at once
                            # Keep small (16-32) since our dataset is small
SEED         = 42           # For reproducibility

# ─── Step 1: Load Dataset & Split into Train / Validation ─────────────────────
# 80% images used for training, 20% for validation (checking model during training)

print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split = 0.2,        # 20% goes to validation
    subset           = "training",
    seed             = SEED,
    image_size       = IMG_SIZE,   # Resize all images to 224x224
    batch_size       = BATCH_SIZE,
    label_mode       = "categorical" # Returns one-hot labels [1,0,0], [0,1,0], [0,0,1]
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split = 0.2,
    subset           = "validation",
    seed             = SEED,
    image_size       = IMG_SIZE,
    batch_size       = BATCH_SIZE,
    label_mode       = "categorical"
)

# ─── Step 2: Check Class Names ─────────────────────────────────────────────────
class_names = train_ds.class_names
print(f"\nClasses detected: {class_names}")
print(f"Expected        : ['high', 'low', 'medium']")
# TensorFlow reads folder names alphabetically

# ─── Step 3: Normalization ─────────────────────────────────────────────────────
# Pixel values are 0-255. We divide by 255 to get 0-1.
# This helps the model train faster and more stably.

normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ─── Step 4: Augmentation (applied only on training data) ─────────────────────
# Validation data is NEVER augmented — we want real performance metrics

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),         # Mirror image left-right
    tf.keras.layers.RandomRotation(0.1),              # Rotate up to 10%
    tf.keras.layers.RandomZoom(0.1),                  # Zoom in/out up to 10%
    tf.keras.layers.RandomBrightness(0.2),            # Brightness variation ±20%
    tf.keras.layers.RandomContrast(0.1),              # Contrast variation ±10%
], name="data_augmentation")

train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# ─── Step 5: Performance Optimization ─────────────────────────────────────────
# prefetch loads next batch while current one is training = faster training

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# ─── Step 6: Verify Everything is Working ─────────────────────────────────────
print("\n=== Dataset Info ===")
for images, labels in train_ds.take(1):
    print(f"  Image batch shape : {images.shape}")   # Should be (16, 224, 224, 3)
    print(f"  Label batch shape : {labels.shape}")   # Should be (16, 3)
    print(f"  Pixel value range : {images.numpy().min():.2f} to {images.numpy().max():.2f}")
    # Should be 0.00 to 1.00 after normalization

# ─── Step 7: Visualize Sample Images ──────────────────────────────────────────
print("\nGenerating sample visualization...")

plt.figure(figsize=(12, 6))
plt.suptitle("Sample Preprocessed Images (Augmented)", fontsize=14)

for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy())
        class_index = np.argmax(labels[i])
        plt.title(class_names[class_index], fontsize=10)
        plt.axis("off")

plt.tight_layout()
plt.savefig("sample_preview.png")
plt.show()
print("\n✅ Preview saved as sample_preview.png")
print("✅ Preprocessing pipeline is ready!")