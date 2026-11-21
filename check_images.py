from pathlib import Path
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "image_dataset"

valid = 0
failed = []

print("Testing images with TensorFlow decoder...")

for path in DATASET_DIR.rglob("*"):
    if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
        continue

    try:
        image = tf.io.read_file(str(path))
        tf.io.decode_image(image, channels=3, expand_animations=False)
        valid += 1

    except Exception as error:
        failed.append(path)
        print("Failed:", path)

print("\nSCAN COMPLETED")
print("TensorFlow-readable images:", valid)
print("Failed images:", len(failed))

with open(BASE_DIR / "tensorflow_failed_images.txt", "w") as f:
    for path in failed:
        f.write(str(path) + "\n")