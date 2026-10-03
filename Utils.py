
from __future__ import annotations
import os
import urllib.request
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
MODEL_PATH = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
LANDMARKS_PER_HAND = 21
VALUES_PER_LANDMARK = 2
FEATURES_PER_HAND = LANDMARKS_PER_HAND * VALUES_PER_LANDMARK
TOTAL_FEATURES = FEATURES_PER_HAND * 2


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),# Thumb  
    (0, 5), (5, 6), (6, 7), (7, 8),# Index Finger
    (5, 9), (9, 10), (10, 11), (11, 12),#Middle Finger
    (9, 13), (13, 14), (14, 15), (15, 16),#Ring Finger
    (13, 17), (17, 18), (18, 19), (19, 20),#Lil Finger
    (0, 17),#Palm Connection
]


def ensure_model(path: str = MODEL_PATH) -> str:
    "Download the MediaPipe Hand Landmarker model if not found"
    if not os.path.isfile(path):
        print(f"Downloading MediaPipe model to: {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
        print("Model download complete.")
    return path


def create_hand_landmarker(
    running_mode: vision.RunningMode = vision.RunningMode.VIDEO,
) -> vision.HandLandmarker:
    """Create a MediaPipe HandLandmarker configured for up to two hands."""
    base_options = mp.tasks.BaseOptions(
        model_asset_path=ensure_model()
    )

    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=running_mode,
        num_hands=2,
        min_hand_detection_confidence=0.65,
        min_hand_presence_confidence=0.65,
        min_tracking_confidence=0.65,
    )

    return vision.HandLandmarker.create_from_options(options)


def to_mp_image(bgr_frame: np.ndarray) -> mp.Image:
    """Convert an OpenCV BGR image to a MediaPipe RGB image."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    rgb = np.ascontiguousarray(rgb)

    return mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )


def _xy_array(hand_landmarks) -> np.ndarray:
    """Return a hand's 21 normalized x/y landmarks as a (21, 2) array."""
    coords = np.asarray(
        [[lm.x, lm.y] for lm in hand_landmarks],
        dtype=np.float32,
    )

    if coords.shape != (LANDMARKS_PER_HAND, 2):
        raise ValueError(
            f"Expected 21 hand landmarks, received shape {coords.shape}."
        )
    return coords


def hands_to_feature_vector(hand_landmarks,handedness,) -> np.ndarray:
    left = None
    right = None
    unknown = []
    count = min(len(hand_landmarks), 2)

    for i in range(count):
        coords = _xy_array(hand_landmarks[i])

        hand_name = ""
        if handedness and i < len(handedness) and handedness[i]:
            hand_name = (
                handedness[i][0].category_name
                or ""
            ).strip().lower()

        if hand_name == "left" and left is None:
            left = coords
        elif hand_name == "right" and right is None:
            right = coords
        else:
            unknown.append(coords)

    if unknown:
        unknown.sort(key=lambda arr: float(arr[0, 0]))

        for coords in unknown:
            if right is None:
                right = coords
            elif left is None:
                left = coords

    present = [hand for hand in (left, right) if hand is not None]

    if not present:
        return np.zeros(TOTAL_FEATURES, dtype=np.float32)

    wrists = np.stack([hand[0] for hand in present])
    origin = wrists.mean(axis=0)

    all_points = np.concatenate(present, axis=0)

    x_span = float(np.ptp(all_points[:, 0]))
    y_span = float(np.ptp(all_points[:, 1]))
    scale = max(x_span, y_span, 1e-6)

    def normalized_or_zeros(hand):
        if hand is None:
            return np.zeros(FEATURES_PER_HAND, dtype=np.float32)

        normalized = (hand - origin) / scale
        return normalized.flatten().astype(np.float32)

    features = np.concatenate(
        [
            normalized_or_zeros(left),
            normalized_or_zeros(right),
        ]
    )

    return features.astype(np.float32)


def feature_vector_length() -> int:
    """Return the fixed feature count used by every project file."""
    return TOTAL_FEATURES


def draw_hand_landmarks(
    frame: np.ndarray,
    hand_landmarks,
    label: str | None = None,
) -> None:
    """Draw one detected hand and an optional Left/Right label."""
    h, w = frame.shape[:2]

    points = [
        (
            int(np.clip(lm.x, 0.0, 1.0) * w),
            int(np.clip(lm.y, 0.0, 1.0) * h),
        )
        for lm in hand_landmarks
    ]

    for start, end in HAND_CONNECTIONS:
        cv2.line(
            frame,
            points[start],
            points[end],
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for point in points:
        cv2.circle(
            frame,
            point,
            4,
            (0, 220, 0),
            -1,
            cv2.LINE_AA,
        )

    if label:
        x, y = points[0]

        cv2.putText(
            frame,
            label,
            (max(5, x - 25), max(25, y - 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
