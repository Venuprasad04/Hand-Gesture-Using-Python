"""
train_model.py

Train a Random Forest classifier from gesture_data.csv.

Outputs:
    gesture_model.joblib
    gesture_labels.joblib
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Utils import feature_vector_length


DATA_FILE = "gesture_data.csv"
MODEL_FILE = "gesture_model.joblib"
LABELS_FILE = "gesture_labels.joblib"

TEST_SIZE = 0.20
RANDOM_STATE = 42


def main() -> None:
    if not os.path.isfile(DATA_FILE):
        raise FileNotFoundError(
            "gesture_data.csv not found. Run data_collection.py first."
        )

    data = pd.read_csv(DATA_FILE)

    expected_feature_columns = [
        f"f{i}"
        for i in range(feature_vector_length())
    ]

    missing = [
        column
        for column in expected_feature_columns + ["label"]
        if column not in data.columns
    ]

    if missing:
        raise ValueError(
            "Dataset format is incorrect. Missing columns: "
            + ", ".join(missing[:10])
        )

    # Keep only exactly the columns this version expects.
    data = data[
        expected_feature_columns + ["label"]
    ].dropna()

    if len(data) == 0:
        raise ValueError("The dataset is empty.")

    counts = data["label"].value_counts()

    if len(counts) < 2:
        raise ValueError(
            "Collect at least two different gesture classes."
        )

    if counts.min() < 5:
        raise ValueError(
            "Every gesture needs at least 5 samples. "
            "Collect substantially more for useful accuracy."
        )

    print()
    print("=" * 60)
    print("DATASET")
    print("=" * 60)
    print(f"Samples : {len(data)}")
    print(f"Gestures: {len(counts)}")
    print()
    print(counts.sort_index().to_string())

    X = data[
        expected_feature_columns
    ].to_numpy(
        dtype=np.float32
    )

    labels = data[
        "label"
    ].astype(str).to_numpy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    model = RandomForestClassifier(
        n_estimators=350,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        oob_score=True,
    )

    print()
    print("Training model...")

    model.fit(
        X_train,
        y_train,
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions,
    )

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(
        f"Test accuracy: {accuracy * 100:.2f}%"
    )
    print(
        f"OOB accuracy : {model.oob_score_ * 100:.2f}%"
    )

    print()
    print("Classification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions,
        )
    )

    joblib.dump(
        model,
        MODEL_FILE,
        compress=3,
    )

    joblib.dump(
        encoder,
        LABELS_FILE,
        compress=3,
    )

    print()
    print(f"Saved: {MODEL_FILE}")
    print(f"Saved: {LABELS_FILE}")


if __name__ == "__main__":
    main()
