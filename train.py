# train.py
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# ─── Suppress oneDNN warnings ─────────────────────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

# ─── Configuration ────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(script_dir, "dataset")
MODEL_PATH   = os.path.join(script_dir, "model", "grey_water_model.h5")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
SEED         = 42
EPOCHS_FROZEN = 15   # Train only our custom layers first
EPOCHS_FINE   = 10   # Then fine-tune top layers of MobileNetV2

# ─── Step 1: Load & Preprocess Dataset ────────────────────────────────────────
print("📂 Loading dataset...")

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

CLASS_NAMES = train_ds.class_names
NUM_CLASSES = len(CLASS_NAMES)
print(f"✅ Classes: {CLASS_NAMES}")

# ─── Step 2: Normalize & Augment ──────────────────────────────────────────────
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

def augment(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.rot90(
        image,
        k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    )
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(normalize, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(augment,   num_parallel_calls=AUTOTUNE)
val_ds   = val_ds.map(normalize,   num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds   = val_ds.prefetch(AUTOTUNE)

# ─── Step 3: Load MobileNetV2 Base (Pre-trained, without top layers) ──────────
# include_top=False means we remove the original 1000-class output layer
# We'll add our own 3-class output layer instead
print("\n🧠 Building model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape = (224, 224, 3),
    include_top = False,          # Remove original classifier
    weights     = "imagenet"      # Use pre-trained ImageNet weights
)

# Freeze the base model — don't change its weights during first training phase
# It already knows how to detect features; we just train our new layers
base_model.trainable = False

# ─── Step 4: Build Full Model ──────────────────────────────────────────────────
inputs = tf.keras.Input(shape=(224, 224, 3))

# Pass through frozen MobileNetV2
x = base_model(inputs, training=False)
# training=False ensures BatchNorm layers stay frozen

# GlobalAveragePooling: converts (7, 7, 1280) feature map → (1280,) vector
x = tf.keras.layers.GlobalAveragePooling2D()(x)

# Dropout: randomly disables 30% of neurons during training
# Forces the model to not rely on any single neuron → reduces overfitting
x = tf.keras.layers.Dropout(0.3)(x)

# Dense layer: learns turbidity-specific patterns from MobileNetV2 features
x = tf.keras.layers.Dense(128, activation='relu')(x)

# Another dropout for extra regularization
x = tf.keras.layers.Dropout(0.2)(x)

# Final output: 3 neurons (one per class), softmax gives probabilities
# Output example: [0.85, 0.10, 0.05] → 85% low, 10% medium, 5% high
outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x)

model = tf.keras.Model(inputs, outputs)

# ─── Step 5: Compile Model ────────────────────────────────────────────────────
# Adam optimizer: adapts learning rate automatically
# categorical_crossentropy: correct loss for multi-class classification
# accuracy: human-readable metric to track
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

model.summary()

# ─── Step 6: Callbacks ────────────────────────────────────────────────────────
# These run automatically after each epoch

callbacks = [
    # Stop training if val_loss doesn't improve for 5 epochs
    tf.keras.callbacks.EarlyStopping(
        monitor  = 'val_loss',
        patience = 5,
        restore_best_weights = True,  # Revert to best weights when stopped
        verbose  = 1
    ),
    # Save the best model automatically
    tf.keras.callbacks.ModelCheckpoint(
        filepath          = MODEL_PATH,
        monitor           = 'val_accuracy',
        save_best_only    = True,
        verbose           = 1
    ),
    # Reduce learning rate if val_loss plateaus
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor  = 'val_loss',
        factor   = 0.5,       # Halve the learning rate
        patience = 3,
        verbose  = 1
    )
]

# ─── Step 7: Phase 1 Training (Frozen base) ────────────────────────────────────
print("\n🚀 Phase 1: Training custom layers (base frozen)...")
history1 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS_FROZEN,
    callbacks       = callbacks
)

# ─── Step 8: Phase 2 — Fine Tuning (Unfreeze top layers) ──────────────────────
# After our custom layers are trained, we slightly update the top
# layers of MobileNetV2 to fine-tune for our specific water images
print("\n🔧 Phase 2: Fine-tuning top layers of MobileNetV2...")

base_model.trainable = True

# Only unfreeze the last 30 layers — don't touch early layers
# Early layers detect basic features (edges) which are universal
# Later layers detect complex patterns which we want to adapt
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with LOWER learning rate — very important!
# High LR would destroy the pre-trained weights we carefully preserved
model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss      = 'categorical_crossentropy',
    metrics   = ['accuracy']
)

history2 = model.fit(
    train_ds,
    validation_data = val_ds,
    epochs          = EPOCHS_FINE,
    callbacks       = callbacks
)

# ─── Step 9: Plot Training History ────────────────────────────────────────────
print("\n📊 Saving training plots...")

# Combine both phases
acc     = history1.history['accuracy']     + history2.history['accuracy']
val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
loss    = history1.history['loss']         + history2.history['loss']
val_loss= history1.history['val_loss']     + history2.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc,     label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.axvline(x=EPOCHS_FROZEN, color='gray', linestyle='--', label='Fine-tune start')
plt.legend()
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss,     label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.axvline(x=EPOCHS_FROZEN, color='gray', linestyle='--', label='Fine-tune start')
plt.legend()
plt.title('Loss')

plt.tight_layout()
plt.savefig("training_history.png")
plt.show()
print("✅ Training plot saved as training_history.png")
print(f"✅ Best model saved to: {MODEL_PATH}")