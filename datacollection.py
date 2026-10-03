from __future__ import annotations
import csv
import os
import time
import cv2
from mediapipe.tasks.python import vision

from Utils import (
    create_hand_landmarker,
    draw_hand_landmarks,
    feature_vector_length,
    hands_to_feature_vector,
    to_mp_image,
)
GESTURES = [
    "i",
    "need",
    "water",
    "love",
    "you",
    "food",
    "help",
    "yes",
    "no",
    "hello",
    "thank_you",
    "please",
]
DATA_FILE = "gesture_data.csv"
SUGGESTED_SAMPLES_PER_GESTURE = 300
SAMPLE_INTERVAL_SECONDS = 0.08
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
def expected_header() -> list[str]:
    return [ f"f{i}" for i in range(feature_vector_length())] + ["label"]


def validate_existing_dataset() -> None:
    if not os.path.isfile(DATA_FILE):
        return

    with open(DATA_FILE, newline="", encoding="utf-8") as file:
        reader = csv.reader(file)
        header = next(reader, None)

    if header != expected_header():
        raise RuntimeError(
            "\nThe existing gesture_data.csv uses a different feature format.\n"
            "Rename or delete it before using this two-hand 84-feature version."
        )


def append_sample(label: str, features) -> None:
    if len(features) != feature_vector_length():
        raise ValueError(
            f"Expected {feature_vector_length()} features, got {len(features)}."
        )

    file_exists = os.path.isfile(DATA_FILE)

    with open(
        DATA_FILE,
        "a",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)

        if not file_exists:
            writer.writerow(expected_header())

        writer.writerow(
            list(features) + [label]
        )


def count_existing_samples() -> dict[str, int]:
    counts = {
        gesture: 0
        for gesture in GESTURES
    }

    if not os.path.isfile(DATA_FILE):
        return counts

    with open(
        DATA_FILE,
        newline="",
        encoding="utf-8",
    ) as file:
        for row in csv.DictReader(file):
            label = row.get("label")
            if label in counts:
                counts[label] += 1

    return counts


def main() -> None:
    validate_existing_dataset()

    cap = cv2.VideoCapture(CAMERA_INDEX)

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam index 0.")

    selected_idx = 0
    recording = False
    counts = count_existing_samples()

    last_sample_time = 0.0

    print()
    print("=" * 60)
    print("HAND GESTURE DATA COLLECTION")
    print("=" * 60)
    print("N     : next gesture")
    print("P     : previous gesture")
    print("SPACE : start / pause recording")
    print("Q     : quit")
    print("=" * 60)

    for i, gesture in enumerate(GESTURES):
        print(f"{i + 1:>2}. {gesture}")

    landmarker = create_hand_landmarker(
        vision.RunningMode.VIDEO
    )

    start_time = time.monotonic()
    last_timestamp = -1

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                print("Could not read camera frame.")
                break
            frame = cv2.flip(frame, 1)

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            if timestamp_ms <= last_timestamp:
                timestamp_ms = last_timestamp + 1

            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                to_mp_image(frame),
                timestamp_ms,
            )

            label_now = GESTURES[selected_idx]
            detected_hands = len(result.hand_landmarks)

            if result.hand_landmarks:
                for i, hand in enumerate(result.hand_landmarks):
                    hand_name = f"Hand {i + 1}"

                    if (
                        result.handedness
                        and i < len(result.handedness)
                        and result.handedness[i]
                    ):
                        hand_name = (
                            result.handedness[i][0].category_name
                            or hand_name
                        )

                    draw_hand_landmarks(
                        frame,
                        hand,
                        hand_name,
                    )

                if recording:
                    now = time.monotonic()

                    if (
                        now - last_sample_time
                        >= SAMPLE_INTERVAL_SECONDS
                    ):
                        features = hands_to_feature_vector(
                            result.hand_landmarks,
                            result.handedness,
                        )

                        append_sample(
                            label_now,
                            features,
                        )

                        counts[label_now] += 1
                        last_sample_time = now

            status = (
                "RECORDING"
                if recording
                else "PAUSED"
            )

            status_color = (
                (0, 0, 255)
                if recording
                else (0, 220, 0)
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 110),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                f"Gesture: {label_now}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                (
                    f"{status} | Samples: "
                    f"{counts[label_now]}/"
                    f"{SUGGESTED_SAMPLES_PER_GESTURE}"
                ),
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                status_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Hands detected: {detected_hands}/2",
                (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Hand Gesture Data Collection",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord(" "):
                recording = not recording

                print(
                    f"{'Recording' if recording else 'Paused'}: "
                    f"{label_now}"
                )

            elif key in (ord("n"), ord("N")):
                selected_idx = (
                    selected_idx + 1
                ) % len(GESTURES)
                recording = False

                print(
                    "Selected:",
                    GESTURES[selected_idx],
                )

            elif key in (ord("p"), ord("P")):
                selected_idx = (
                    selected_idx - 1
                ) % len(GESTURES)
                recording = False

                print(
                    "Selected:",
                    GESTURES[selected_idx],
                )

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()

        print()
        print("Final sample counts:")
        for gesture, count in counts.items():
            print(f"  {gesture:<15} {count}")


if __name__ == "__main__":
    main()
