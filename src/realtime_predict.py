import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf


LABELS = ["evet", "hayir", "tamam"]
SEQUENCE_LENGTH = 20

MODEL_PATH = "models/lip_reading_model.keras"

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
    model = tf.keras.models.load_model(MODEL_PATH)

    cap = cv2.VideoCapture(0)

    prediction_text = "SPACE ile test et"
    confidence_text = ""

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

            cv2.putText(frame, f"Tahmin: {prediction_text}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.putText(frame, f"Guven: {confidence_text}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            cv2.putText(frame, "SPACE = kelime test | Q = cikis", (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                x_min, y_min, x_max, y_max = get_mouth_box(face_landmarks, w, h)

                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

            cv2.imshow("Realtime Lip Reading", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord(" "):
                sequence = []

                print("[INFO] Test kaydi basladi. Kelimeyi soyle.")

                while len(sequence) < SEQUENCE_LENGTH:
                    ret, frame = cap.read()

                    if not ret:
                        break

                    frame = cv2.flip(frame, 1)
                    h, w, _ = frame.shape

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = face_mesh.process(rgb_frame)

                    if results.multi_face_landmarks:
                        face_landmarks = results.multi_face_landmarks[0]
                        x_min, y_min, x_max, y_max = get_mouth_box(face_landmarks, w, h)

                        mouth_crop = frame[y_min:y_max, x_min:x_max]

                        if mouth_crop.size > 0:
                            mouth_crop = cv2.resize(mouth_crop, (100, 50))
                            mouth_crop = cv2.cvtColor(mouth_crop, cv2.COLOR_BGR2RGB)
                            mouth_crop = mouth_crop / 255.0

                            sequence.append(mouth_crop)

                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            cv2.putText(frame, f"Kaydediliyor: {len(sequence)}/{SEQUENCE_LENGTH}",
                                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                        (0, 0, 255), 2)

                            cv2.imshow("Realtime Lip Reading", frame)
                            cv2.waitKey(1)

                if len(sequence) == SEQUENCE_LENGTH:
                    input_data = np.expand_dims(sequence, axis=0)
                    predictions = model.predict(input_data, verbose=0)

                    predicted_index = np.argmax(predictions[0])
                    confidence = predictions[0][predicted_index]

                    prediction_text = LABELS[predicted_index]
                    confidence_text = f"{confidence:.2f}"

                    print(f"[TAHMIN] {prediction_text} - Guven: {confidence_text}")

            elif key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()