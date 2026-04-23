# save as check_dataset.py and run it
import os

dataset_path = "dataset"
classes = ["low", "medium", "high"]

print("=== Dataset Summary ===")
total = 0
for cls in classes:
    path = os.path.join(dataset_path, cls)
    count = len([f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
    total += count
    status = "✅" if count >= 50 else "⚠️  Need more images"
    print(f"  {cls.upper():10s}: {count:4d} images  {status}")

print(f"\n  TOTAL     : {total:4d} images")
print(f"\n  Minimum needed: 150 | You have: {total}")