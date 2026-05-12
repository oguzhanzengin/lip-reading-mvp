import os
import cv2
import json
import numpy as np


DATA_DIR = "data/external_raw"
OUTPUT_DIR = "data/external_processed"

SEQUENCE_LENGTH = 20

labels = sorted([
    item for item in os.listdir(DATA_DIR)
    if os.path.isdir(os.path.join(DATA_DIR, item))
])

LABELS = {label: idx for idx, label in enumerate(labels)}

X = []
y = []

for label_name, label_index in LABELS.items():
    label_path = os.path.join(DATA_DIR, label_name)

    sequences = sorted(os.listdir(label_path))

    for seq in sequences:
        seq_path = os.path.join(label_path, seq)

        if not os.path.isdir(seq_path):
            continue

        frames = []

        frame_files = sorted(
            os.listdir(seq_path),
            key=lambda x: int(x.split(".")[0])
        )

        for frame_file in frame_files:
            frame_path = os.path.join(seq_path, frame_file)

            img = cv2.imread(frame_path)

            if img is None:
                continue

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img / 255.0

            frames.append(img)

        if len(frames) == SEQUENCE_LENGTH:
            X.append(frames)
            y.append(label_index)

X = np.array(X, dtype=np.float32)
y = np.array(y)

os.makedirs(OUTPUT_DIR, exist_ok=True)

np.save(os.path.join(OUTPUT_DIR, "X.npy"), X)
np.save(os.path.join(OUTPUT_DIR, "y.npy"), y)

with open(os.path.join(OUTPUT_DIR, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(LABELS, f, ensure_ascii=False, indent=4)

print("LABELS:", LABELS)
print("X shape:", X.shape)
print("y shape:", y.shape)
print("[✓] External dataset kaydedildi.")