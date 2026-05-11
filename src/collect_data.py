import os
import cv2
import time
import numpy as np
import mediapipe as mp


LABELS = ["evet", "hayir", "tamam"]
SEQUENCE_LENGTH = 20


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


def create_directories():
    for label in LABELS:
        os.makedirs(f"data/raw/{label}", exist_ok=True)


def save_sequence(sequence, label):
    existing = len(os.listdir(f"data/raw/{label}"))
    sequence_path = f"data/raw/{label}/seq_{existing}"

    os.makedirs(sequence_path)

    for idx, frame in enumerate(sequence):
        cv2.imwrite(
            f"{sequence_path}/{idx}.jpg",
            frame
        )

    print(f"[✓] Kaydedildi: {sequence_path}")


def main():
    create_directories()

    current_label_index = 0

    cap = cv2.VideoCapture(0)

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

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            label = LABELS[current_label_index]

            cv2.putText(
                frame,
                f"Kelime: {label}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "SPACE = kayit | N = sonraki kelime | Q = cikis",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            mouth_crop = None

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]

                x_min, y_min, x_max, y_max = get_mouth_box(
                    face_landmarks,
                    w,
                    h
                )

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                mouth_crop = frame[y_min:y_max, x_min:x_max]

            cv2.imshow("Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):

                sequence = []

                for countdown in range(3, 0, -1):

                    temp_frame = frame.copy()

                    cv2.putText(
                        temp_frame,
                        str(countdown),
                        (300, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,
                        (0, 0, 255),
                        5
                    )

                    cv2.imshow("Collector", temp_frame)
                    cv2.waitKey(1000)

                print(f"[INFO] Kayit basladi: {label}")

                while len(sequence) < SEQUENCE_LENGTH:

                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)

                    rgb_frame = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:

                        face_landmarks = results.multi_face_landmarks[0]

                        x_min, y_min, x_max, y_max = get_mouth_box(
                            face_landmarks,
                            w,
                            h
                        )

                        mouth_crop = frame[
                            y_min:y_max,
                            x_min:x_max
                        ]

                        if mouth_crop.size > 0:

                            mouth_crop = cv2.resize(
                                mouth_crop,
                                (100, 50)
                            )

                            sequence.append(mouth_crop)

                            cv2.imshow(
                                "Mouth Recording",
                                mouth_crop
                            )

                            cv2.waitKey(1)

                if len(sequence) == SEQUENCE_LENGTH:
                    save_sequence(sequence, label)

            elif key == ord("n"):
                current_label_index = (
                    current_label_index + 1
                ) % len(LABELS)

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()