# evaluate.py
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

# ─── Suppress warnings ────────────────────────────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

# ─── Configuration ────────────────────────────────────────────────────────────
script_dir   = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(script_dir, "dataset")
MODEL_PATH   = os.path.join(script_dir, "model", "grey_water_model.h5")
IMG_SIZE     = (224, 224)
BATCH_SIZE   = 16
SEED         = 42

# ─── Step 1: Load Validation Dataset ──────────────────────────────────────────
print("📂 Loading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    validation_split = 0.2,
    subset           = "validation",
    seed             = SEED,
    image_size       = IMG_SIZE,
    batch_size       = BATCH_SIZE,
    label_mode       = "categorical"
)

CLASS_NAMES = val_ds.class_names
print(f"✅ Classes: {CLASS_NAMES}")

# ─── Step 2: Normalize ────────────────────────────────────────────────────────
def normalize(image, label):
    image = tf.cast(image, tf.float32) / 255.0
    return image, label

val_ds = val_ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

# ─── Step 3: Load Saved Model ─────────────────────────────────────────────────
print(f"\n🧠 Loading model from: {MODEL_PATH}")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

# ─── Step 4: Get Predictions ──────────────────────────────────────────────────
print("\n🔍 Running predictions on validation set...")

all_true  = []   # actual labels
all_pred  = []   # predicted labels
all_probs = []   # prediction confidence scores

for images, labels in val_ds:
    predictions = model.predict(images, verbose=0)
    true_classes = np.argmax(labels.numpy(), axis=1)
    pred_classes = np.argmax(predictions, axis=1)

    all_true.extend(true_classes)
    all_pred.extend(pred_classes)
    all_probs.extend(np.max(predictions, axis=1))

all_true  = np.array(all_true)
all_pred  = np.array(all_pred)
all_probs = np.array(all_probs)

# ─── Step 5: Overall Accuracy ─────────────────────────────────────────────────
accuracy = np.mean(all_true == all_pred) * 100
print(f"\n{'='*45}")
print(f"  Overall Validation Accuracy: {accuracy:.2f}%")
print(f"{'='*45}")

# ─── Step 6: Classification Report ───────────────────────────────────────────
print("\n📋 Classification Report:")
print(classification_report(
    all_true,
    all_pred,
    target_names = CLASS_NAMES,
    digits       = 3
))

# ─── Step 7: Confusion Matrix ─────────────────────────────────────────────────
print("📊 Generating confusion matrix...")

cm = confusion_matrix(all_true, all_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot      = True,
    fmt        = 'd',
    cmap       = 'Blues',
    xticklabels = CLASS_NAMES,
    yticklabels = CLASS_NAMES,
    linewidths  = 0.5
)
plt.title("Confusion Matrix — Grey Water Classifier", fontsize=14, pad=15)
plt.ylabel("Actual Label",    fontsize=12)
plt.xlabel("Predicted Label", fontsize=12)
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
print("✅ Confusion matrix saved as confusion_matrix.png")

# ─── Step 8: Confidence Analysis ──────────────────────────────────────────────
print(f"\n📈 Confidence Analysis:")
print(f"  Average confidence : {all_probs.mean()*100:.1f}%")
print(f"  Min confidence     : {all_probs.min()*100:.1f}%")
print(f"  Max confidence     : {all_probs.max()*100:.1f}%")

# ─── Step 9: Per-Image Prediction Detail ──────────────────────────────────────
print(f"\n🔎 Per-Prediction Breakdown:")
print(f"  {'#':<4} {'Actual':<10} {'Predicted':<10} {'Confidence':<12} {'Result'}")
print(f"  {'-'*50}")
for i, (true, pred, prob) in enumerate(zip(all_true, all_pred, all_probs)):
    result = "✅" if true == pred else "❌"
    print(f"  {i+1:<4} {CLASS_NAMES[true]:<10} {CLASS_NAMES[pred]:<10} {prob*100:<12.1f} {result}")

# ─── Step 10: Check if Model is Genuinely Learning ────────────────────────────
print(f"\n{'='*45}")
print("  Model Health Check:")
print(f"{'='*45}")

unique_preds = len(set(all_pred))
if unique_preds == 1:
    print("  ⚠️  WARNING: Model predicts only ONE class!")
    print("      It is NOT learning — just guessing the majority class.")
elif accuracy < 50:
    print("  ⚠️  WARNING: Accuracy below 50% — worse than random guessing.")
    print("      Check your dataset labels and class balance.")
elif accuracy < 70:
    print("  ⚠️  FAIR: Model is learning but needs more data.")
elif accuracy < 90:
    print("  ✅  GOOD: Model is learning well.")
else:
    print("  🎉 EXCELLENT: Model is performing very well!")

print(f"  Classes predicted  : {[CLASS_NAMES[p] for p in sorted(set(all_pred))]}")
print(f"  Unique predictions : {unique_preds} out of {len(CLASS_NAMES)} classes")
print(f"{'='*45}\n")