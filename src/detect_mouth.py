import cv2
import mediapipe as mp


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


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Kameradan görüntü alınamadı.")
                break

            frame = cv2.flip(frame, 1)

            h, w, _ = frame.shape
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                x_min, y_min, x_max, y_max = get_mouth_box(face_landmarks, w, h)

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2
                )

                mouth_crop = frame[y_min:y_max, x_min:x_max]

                if mouth_crop.size > 0:
                    cv2.imshow("Mouth Crop", mouth_crop)

            cv2.imshow("Lip Reading MVP - Camera", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()