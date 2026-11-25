from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

TARGET_RATIO = 9 / 16


def find_speaker_crop(
    video: Path,
    start: float,
    end: float,
    *,
    samples: int = 15,
) -> tuple[int, int, int, int] | None:
    cascade_path = _cascade_path()
    if cascade_path is None:
        return None
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None

    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        return None
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_width = int(round(height * TARGET_RATIO))
    crop_width -= crop_width % 2
    capture.release()
    if width <= 0 or height <= 0 or crop_width >= width:
        return None

    duration = end - start
    if duration <= 0.5:
        return None
    count = max(3, min(samples, int(duration * 2)))
    times = np.linspace(start + min(0.5, duration / 4), max(start, end - min(0.5, duration / 4)), count)

    capture = cv2.VideoCapture(str(video))
    detections: list[tuple[float, float]] = []
    min_size = max(40, int(height * 0.08))
    for timestamp in times:
        capture.set(cv2.CAP_PROP_POS_MSEC, float(timestamp) * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_size, min_size),
            maxSize=(int(width * 0.5), int(height * 0.5)),
        )
        if len(faces) == 0:
            continue
        x, y, face_w, face_h = max(faces, key=lambda face: float(face[2] * face[3]))
        weight = float(face_w * face_h) / float(width * height)
        if y > height * 0.65:
            weight *= 0.6
        detections.append((float(x + face_w / 2), weight))
    capture.release()
    if not detections:
        return None

    bucket_size = max(1.0, crop_width * 0.4)
    buckets: dict[int, list[tuple[float, float]]] = {}
    for center_x, weight in detections:
        buckets.setdefault(int(center_x // bucket_size), []).append((center_x, weight))
    best = max(buckets, key=lambda key: sum(weight for _, weight in buckets[key]))
    group = buckets[best]
    total = sum(weight for _, weight in group)
    center = sum(position * weight for position, weight in group) / total
    x = int(round(center - crop_width / 2))
    x = max(0, min(width - crop_width, x))
    return x, 0, crop_width, height


def _cascade_path() -> Path | None:
    candidates: list[Path] = []
    try:
        import cv2.data

        candidates.append(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
    except (ImportError, AttributeError):
        pass
    candidates.append(Path("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"))
    candidates.append(Path("/usr/share/obs/obs-plugins/advanced-scene-switcher/res/cascadeClassifiers/haarcascade_frontalface_default.xml"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None
