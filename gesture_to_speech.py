
from __future__ import annotations
import queue
import threading
import time

import cv2
import joblib
import numpy as np
from mediapipe.tasks.python import vision

from Utils import (
    create_hand_landmarker,
    draw_hand_landmarks,
    feature_vector_length,
    hands_to_feature_vector,
    to_mp_image,
)


MODEL_FILE = "gesture_model.joblib"
LABELS_FILE = "gesture_labels.joblib"

CONFIDENCE_THRESHOLD = 0.70
STABLE_FRAMES_REQUIRED = 5
NO_HAND_RESET_FRAMES = 5

CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480


class SpeechWorker:
    """Run Windows SAPI speech on a background thread."""

    def __init__(self) -> None:
        self.q = queue.Queue()

        self.thread = threading.Thread(
            target=self._run,
            daemon=True,
        )

        self.thread.start()

    def _run(self) -> None:
        import pythoncom
        import win32com.client

        pythoncom.CoInitialize()

        try:
            speaker = win32com.client.Dispatch(
                "SAPI.SpVoice"
            )

            speaker.Rate = 0
            speaker.Volume = 100

            print("Windows speech engine started.")

            while True:
                text = self.q.get()

                try:
                    if text is None:
                        return

                    speaker.Speak(text)

                except Exception as error:
                    print("Speech error:", error)

                finally:
                    self.q.task_done()

        finally:
            pythoncom.CoUninitialize()

    def say(self, text: str) -> None:
        text = text.strip()

        if text:
            self.q.put(text)

    def stop(self) -> None:
        self.q.put(None)
        self.thread.join(timeout=3)


def sentence_text(words: list[str]) -> str:
    if not words:
        return ""

    return " ".join(
        word.replace("_", " ")
        for word in words
    ).capitalize()


def main() -> None:
    try:
        model = joblib.load(
            MODEL_FILE
        )
        encoder = joblib.load(
            LABELS_FILE
        )

    except FileNotFoundError:
        print(
            "gesture_model.joblib or gesture_labels.joblib "
            "was not found. Run train_model.py first."
        )
        return

    if getattr(model, "n_features_in_", None) != feature_vector_length():
        print(
            "Model feature size does not match this version.\n"
            "Recollect data and retrain the model using the new files."
        )
        return

    print(
        "Available gestures:",
        list(encoder.classes_),
    )

    speech = SpeechWorker()

    cap = cv2.VideoCapture(
        CAMERA_INDEX
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH,
    )
    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT,
    )

    if not cap.isOpened():
        print("Could not open camera.")
        speech.stop()
        return

    landmarker = create_hand_landmarker(
        vision.RunningMode.VIDEO
    )

    start_time = time.monotonic()
    last_timestamp = -1

    sentence_words: list[str] = []

    candidate_label = None
    candidate_frames = 0

    # Stops a held gesture from being inserted over and over.
    last_added_label = None

    no_hand_frames = 0

    try:
        while True:
            ok, frame = cap.read()

            if not ok:
                break

            frame = cv2.flip(
                frame,
                1,
            )

            timestamp_ms = int(
                (time.monotonic() - start_time)
                * 1000
            )

            if timestamp_ms <= last_timestamp:
                timestamp_ms = (
                    last_timestamp + 1
                )

            last_timestamp = timestamp_ms

            result = landmarker.detect_for_video(
                to_mp_image(frame),
                timestamp_ms,
            )

            display_prediction = "Show a gesture"
            confidence = 0.0

            if result.hand_landmarks:
                no_hand_frames = 0

                # Draw all hands.
                for i, hand in enumerate(
                    result.hand_landmarks
                ):
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

                # One combined 84-feature representation.
                features = hands_to_feature_vector(
                    result.hand_landmarks,
                    result.handedness,
                )

                sample = np.asarray(
                    features,
                    dtype=np.float32,
                ).reshape(
                    1,
                    -1,
                )

                probabilities = (
                    model.predict_proba(
                        sample
                    )[0]
                )

                best_index = int(
                    np.argmax(
                        probabilities
                    )
                )

                confidence = float(
                    probabilities[
                        best_index
                    ]
                )

                encoded_class = int(
                    model.classes_[
                        best_index
                    ]
                )

                label = encoder.inverse_transform(
                    [encoded_class]
                )[0]

                display_prediction = (
                    f"{label.replace('_', ' ')} "
                    f"{confidence * 100:.1f}%"
                )

                if (
                    confidence
                    >= CONFIDENCE_THRESHOLD
                ):
                    if label == candidate_label:
                        candidate_frames += 1
                    else:
                        candidate_label = label
                        candidate_frames = 1

                    if (
                        candidate_frames
                        >= STABLE_FRAMES_REQUIRED
                    ):
                        if (
                            label
                            != last_added_label
                        ):
                            sentence_words.append(
                                label
                            )

                            last_added_label = label

                            print(
                                "Added:",
                                label,
                                "| Sentence:",
                                sentence_text(
                                    sentence_words
                                ),
                            )

                        candidate_frames = (
                            STABLE_FRAMES_REQUIRED
                        )

                else:
                    candidate_label = None
                    candidate_frames = 0

            else:
                no_hand_frames += 1

                candidate_label = None
                candidate_frames = 0

                # Removing hands briefly allows the same word
                # to be entered again later.
                if (
                    no_hand_frames
                    >= NO_HAND_RESET_FRAMES
                ):
                    last_added_label = None

            current_sentence = sentence_text(
                sentence_words
            )

            cv2.rectangle(
                frame,
                (0, 0),
                (frame.shape[1], 145),
                (0, 0, 0),
                -1,
            )

            cv2.putText(
                frame,
                "Gesture -> Sentence -> Speech",
                (10, 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                f"Gesture: {display_prediction}",
                (10, 61),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.66,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            shown_sentence = (
                current_sentence
                if current_sentence
                else "-"
            )

            cv2.putText(
                frame,
                f"Sentence: {shown_sentence}",
                (10, 96),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.62,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                frame,
                "S=Speak  C=Clear  B=Back  Q=Quit",
                (10, 128),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.52,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(
                "Gesture To Speech",
                frame,
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                ord("Q"),
            ):
                break

            if key in (
                ord("s"),
                ord("S"),
            ):
                text = sentence_text(
                    sentence_words
                )

                if text:
                    print("Speaking:", text)
                    speech.say(text)

            elif key in (
                ord("c"),
                ord("C"),
            ):
                sentence_words.clear()
                candidate_label = None
                candidate_frames = 0
                last_added_label = None

                print("Sentence cleared.")

            elif key in (
                ord("b"),
                ord("B"),
            ):
                if sentence_words:
                    removed = sentence_words.pop()
                    last_added_label = None

                    print(
                        "Removed:",
                        removed,
                    )

    finally:
        landmarker.close()
        cap.release()
        cv2.destroyAllWindows()
        speech.stop()

        print("Program closed.")


if __name__ == "__main__":
    main()
