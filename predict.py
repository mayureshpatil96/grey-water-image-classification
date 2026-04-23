# predict.py
# Use this to test the model on ANY new image

import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image

# ─── Suppress warnings ────────────────────────────────────────────────────────
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '2'

# ─── Configuration ────────────────────────────────────────────────────────────
script_dir  = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(script_dir, "model", "grey_water_model.h5")
IMG_SIZE    = (224, 224)
CLASS_NAMES = ['high', 'low', 'medium']  # Alphabetical — must match training order

# Turbidity descriptions shown to user
CLASS_INFO = {
    'low'    : ('🟢 LOW TURBIDITY',    'Water is clear. Safe for most grey water reuse.'),
    'medium' : ('🟡 MEDIUM TURBIDITY', 'Water is cloudy. Suitable for irrigation only.'),
    'high'   : ('🔴 HIGH TURBIDITY',   'Water is very murky. Needs treatment before reuse.')
}

# ─── Step 1: Load Model ───────────────────────────────────────────────────────
print("🧠 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded!\n")

# ─── Step 2: Get Image Path ───────────────────────────────────────────────────
# You can pass image path as argument: python predict.py myimage.jpg
# Or it will ask you to type the path

if len(sys.argv) > 1:
    image_path = sys.argv[1]
else:
    image_path = input("📁 Enter full path to image: ").strip().strip('"')

# Validate file exists
if not os.path.exists(image_path):
    print(f"❌ File not found: {image_path}")
    sys.exit(1)

print(f"📸 Processing: {os.path.basename(image_path)}")

# ─── Step 3: Load & Preprocess Image ─────────────────────────────────────────
# Same preprocessing as training — MUST match exactly
img = Image.open(image_path).convert('RGB')   # Ensure 3 channels (no alpha)
img_resized = img.resize(IMG_SIZE)             # Resize to 224x224

# Convert to numpy array and normalize to 0-1
img_array = np.array(img_resized, dtype=np.float32) / 255.0

# Add batch dimension: (224,224,3) → (1,224,224,3)
# Model expects a batch, even for single image
img_batch = np.expand_dims(img_array, axis=0)

# ─── Step 4: Predict ──────────────────────────────────────────────────────────
predictions = model.predict(img_batch, verbose=0)

# predictions shape: [[0.02, 0.91, 0.07]]
# Each value = probability for [high, low, medium]

predicted_index = np.argmax(predictions[0])
predicted_class = CLASS_NAMES[predicted_index]
confidence      = predictions[0][predicted_index] * 100

# ─── Step 5: Display Results ──────────────────────────────────────────────────
label, description = CLASS_INFO[predicted_class]

print(f"\n{'='*45}")
print(f"  PREDICTION RESULT")
print(f"{'='*45}")
print(f"  {label}")
print(f"  {description}")
print(f"{'='*45}")
print(f"\n  Confidence Scores:")
for i, cls in enumerate(CLASS_NAMES):
    bar = '█' * int(predictions[0][i] * 20)
    print(f"  {cls:<8}: {predictions[0][i]*100:5.1f}%  {bar}")

# ─── Step 6: Visual Output ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Left: Original image
axes[0].imshow(img)
axes[0].set_title("Input Image", fontsize=12)
axes[0].axis("off")

# Right: Confidence bar chart
colors = ['#e74c3c', '#2ecc71', '#f39c12']  # red, green, orange
bars = axes[1].bar(CLASS_NAMES, predictions[0] * 100, color=colors, edgecolor='black')
axes[1].set_ylim(0, 110)
axes[1].set_ylabel("Confidence (%)", fontsize=11)
axes[1].set_title("Prediction Confidence", fontsize=12)

# Add value labels on bars
for bar, val in zip(bars, predictions[0] * 100):
    axes[1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 2,
        f'{val:.1f}%',
        ha='center', fontsize=11, fontweight='bold'
    )

# Highlight predicted class
plt.suptitle(f"Result: {label}", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("prediction_result.png")
plt.show()
print("\n✅ Result saved as prediction_result.png")