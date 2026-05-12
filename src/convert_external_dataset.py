import os
import cv2
import shutil
import numpy as np


SOURCE_DIR = "external_datasets"
TARGET_DIR = "data/external_raw"

SEQUENCE_LENGTH = 20
IMG_SIZE = (100, 50)  # width, height


def reset_target_dir():
    if os.path.exists(TARGET_DIR):
        shutil.rmtree(TARGET_DIR)

    os.makedirs(TARGET_DIR, exist_ok=True)


def load_frames(sequence_path):
    frame_files = sorted([
        f for f in os.listdir(sequence_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    frames = []

    for frame_file in frame_files:
        frame_path = os.path.join(sequence_path, frame_file)
        img = cv2.imread(frame_path)

        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        frames.append(img)

    return frames


def resample_frames(frames, target_length=SEQUENCE_LENGTH):
    if len(frames) == 0:
        return []

    indices = np.linspace(0, len(frames) - 1, target_length).astype(int)
    return [frames[i] for i in indices]


def save_sequence(frames, label, sequence_index):
    target_sequence_dir = os.path.join(
        TARGET_DIR,
        label,
        f"seq_{sequence_index}"
    )

    os.makedirs(target_sequence_dir, exist_ok=True)

    for idx, frame in enumerate(frames):
        cv2.imwrite(
            os.path.join(target_sequence_dir, f"{idx}.jpg"),
            frame
        )


def main():
    reset_target_dir()

    labels = sorted([
        item for item in os.listdir(SOURCE_DIR)
        if os.path.isdir(os.path.join(SOURCE_DIR, item))
    ])

    print("Bulunan label'lar:")
    for label in labels:
        print("-", label)

    total_sequences = 0

    for label in labels:
        label_path = os.path.join(SOURCE_DIR, label)

        sequence_dirs = sorted([
            item for item in os.listdir(label_path)
            if os.path.isdir(os.path.join(label_path, item))
        ])

        saved_count = 0

        for seq_dir in sequence_dirs:
            sequence_path = os.path.join(label_path, seq_dir)

            frames = load_frames(sequence_path)

            if len(frames) == 0:
                continue

            frames = resample_frames(frames, SEQUENCE_LENGTH)

            if len(frames) == SEQUENCE_LENGTH:
                save_sequence(frames, label, saved_count)
                saved_count += 1
                total_sequences += 1

        print(f"[✓] {label}: {saved_count} sequence")

    print(f"\n[✓] Toplam sequence: {total_sequences}")
    print(f"[✓] Kaydedildi: {TARGET_DIR}")


if __name__ == "__main__":
    main()