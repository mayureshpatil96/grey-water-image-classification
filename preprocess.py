import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

# ─── Configuration ────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(script_dir, "dataset")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
SEED         = 42

# ─── Load Dataset ─────────────────────────────────────────────────────────────
print("Loading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split = 0.2,
    subset           = "training",
    seed             = SEED,
    image_size       = IMG_SIZE,
    batch_size       = BATCH_SIZE,
    label_mode       = "categorical"
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

class_names = train_ds.class_names
print(f"Classes detected: {class_names}")

# ─── Augmentation (BEFORE normalization, on 0-255 range) ──────────────────────
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
], name="data_augmentation")
# NOTE: Removed RandomBrightness & RandomContrast here
# They will be added INSIDE the model safely in Step 4

# ─── Normalization (AFTER augmentation, converts 0-255 → 0-1) ─────────────────
normalization_layer = tf.keras.layers.Rescaling(1.0 / 255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(data_augmentation(x, training=True)), y))
val_ds   = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ─── Performance Optimization ──────────────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds   = val_ds.prefetch(buffer_size=AUTOTUNE)

# ─── Verify ───────────────────────────────────────────────────────────────────
print("\n=== Dataset Info ===")
for images, labels in train_ds.take(1):
    print(f"  Image batch shape : {images.shape}")
    print(f"  Label batch shape : {labels.shape}")
    pixel_min = images.numpy().min()
    pixel_max = images.numpy().max()
    print(f"  Pixel value range : {pixel_min:.2f} to {pixel_max:.2f}")
    if pixel_max <= 1.0:
        print("  Normalization     : ✅ Correct (0 to 1)")
    else:
        print("  Normalization     : ❌ Still wrong!")

# ─── Visualize ────────────────────────────────────────────────────────────────
print("\nGenerating sample visualization...")
plt.figure(figsize=(12, 6))
plt.suptitle("Sample Preprocessed Images (Augmented)", fontsize=14)

for images, labels in train_ds.take(1):
    for i in range(min(9, images.shape[0])):
        ax = plt.subplot(3, 3, i + 1)
        # Clip to [0,1] just for display safety
        img = np.clip(images[i].numpy(), 0.0, 1.0)
        plt.imshow(img)
        class_index = np.argmax(labels[i])
        plt.title(class_names[class_index], fontsize=10)
        plt.axis("off")

plt.tight_layout()
plt.savefig("sample_preview.png")
plt.show()
print("\n✅ Preview saved as sample_preview.png")
print("✅ Preprocessing pipeline is ready!")