import argparse
import os
import cv2
import mediapipe as mp


CLIPS_DIR = "auto_dataset/clips"
MOUTH_CROPS_DIR = "auto_dataset/mouth_crops"

IMG_SIZE = (100, 50)  # width, height

mp_face_mesh = mp.solutions.face_mesh


def get_mouth_box(face_landmarks, frame_width, frame_height):
    mouth_indices = [
        61, 146, 91, 181, 84, 17,
        314, 405, 321, 375, 291,
        78, 95, 88, 178, 87, 14,
        317, 402, 318, 324, 308
    ]

    x_points = []
    y_points = []

    for idx in mouth_indices:
        landmark = face_landmarks.landmark[idx]
        x_points.append(int(landmark.x * frame_width))
        y_points.append(int(landmark.y * frame_height))

    x_min = max(min(x_points) - 30, 0)
    y_min = max(min(y_points) - 30, 0)
    x_max = min(max(x_points) + 30, frame_width)
    y_max = min(max(y_points) + 30, frame_height)

    return x_min, y_min, x_max, y_max


def extract_from_clip(clip_path, output_dir, max_frames=None):
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(clip_path)

    saved_frames = 0
    total_frames = 0
    detected_frames = 0

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()

            if not ret:
                break

            total_frames += 1

            if max_frames is not None and saved_frames >= max_frames:
                break

            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                continue

            detected_frames += 1

            face_landmarks = results.multi_face_landmarks[0]

            x_min, y_min, x_max, y_max = get_mouth_box(
                face_landmarks,
                w,
                h
            )

            mouth_crop = frame[y_min:y_max, x_min:x_max]

            if mouth_crop.size == 0:
                continue

            mouth_crop = cv2.resize(mouth_crop, IMG_SIZE)

            output_path = os.path.join(
                output_dir,
                f"{saved_frames:04d}.jpg"
            )

            cv2.imwrite(output_path, mouth_crop)

            saved_frames += 1

    cap.release()

    quality_score = detected_frames / total_frames if total_frames > 0 else 0

    return {
        "total_frames": total_frames,
        "detected_frames": detected_frames,
        "saved_frames": saved_frames,
        "quality_score": quality_score,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--clips-dir",
        default=CLIPS_DIR,
        help="Directory containing mp4 clips"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional max frames per clip"
    )

    args = parser.parse_args()

    os.makedirs(MOUTH_CROPS_DIR, exist_ok=True)

    clip_files = sorted([
        f for f in os.listdir(args.clips_dir)
        if f.lower().endswith(".mp4")
    ])

    print(f"[INFO] Bulunan clip sayısı: {len(clip_files)}")

    for clip_file in clip_files:
        clip_path = os.path.join(args.clips_dir, clip_file)
        clip_id = os.path.splitext(clip_file)[0]

        output_dir = os.path.join(MOUTH_CROPS_DIR, clip_id)

        stats = extract_from_clip(
            clip_path=clip_path,
            output_dir=output_dir,
            max_frames=args.max_frames
        )

        print(
            f"[✓] {clip_id} | "
            f"frames={stats['total_frames']} | "
            f"detected={stats['detected_frames']} | "
            f"saved={stats['saved_frames']} | "
            f"quality={stats['quality_score']:.2f}"
        )


if __name__ == "__main__":
    main()