"""Convert one pre-trimmed ASL video into SignLab landmark JSON."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

HAND_POINTS = 21
POSE_INDICES = (11, 12, 13, 14, 23, 24)
MAX_SECONDS = 8
SAMPLE_FPS = 24


def coordinates(landmark):
    """Preserve image-normalized coordinates, including points outside the image."""
    if landmark is None:
        return [None, None, None]
    values = (landmark.x, landmark.y, landmark.z)
    if any(value is None or not math.isfinite(value) for value in values):
        return [None, None, None]
    return [round(value, 6) for value in values]


def frame_landmarks(result):
    hands = []
    for hand in (result.left_hand_landmarks, result.right_hand_landmarks):
        if len(hand) == HAND_POINTS:
            hands.extend(coordinates(point) for point in hand)
        else:
            hands.extend(coordinates(None) for _ in range(HAND_POINTS))

    pose = result.pose_landmarks
    return hands + [
        coordinates(pose[index] if index < len(pose) else None)
        for index in POSE_INDICES
    ]


def extract(video: Path, model: Path) -> list:
    # Keep video dependencies optional for the rest of SignLab.
    import cv2
    import mediapipe as mp

    capture = cv2.VideoCapture(str(video))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not open the video.")
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or not 0 < fps <= 1000:
            raise ValueError("Could not read a usable frame rate. Try a constant-frame-rate MP4.")

        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
        )
        frames = []
        visible_hand_frames = 0
        index = 0
        with mp.tasks.vision.HolisticLandmarker.create_from_options(options) as landmarker:
            while True:
                decoded, image = capture.read()
                if not decoded:
                    break
                seconds = index / fps
                index += 1
                if seconds >= MAX_SECONDS:
                    raise ValueError(f"Trim the recording to one sign and at most {MAX_SECONDS} seconds.")
                if seconds + 1e-9 < len(frames) / SAMPLE_FPS:
                    continue

                rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                result = landmarker.detect_for_video(mp_image, round(seconds * 1000))
                landmarks = frame_landmarks(result)
                frames.append(landmarks)
                visible_hand_frames += any(
                    point[0] is not None for point in landmarks[: 2 * HAND_POINTS]
                )
    finally:
        capture.release()

    if visible_hand_frames < 2:
        raise ValueError(
            "Detected hands in fewer than two frames. Use a well-lit clip with hands visible."
        )
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="One pre-trimmed sign, up to eight seconds.")
    parser.add_argument("--model", type=Path, required=True, help="Local holistic_landmarker.task.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for landmark JSON.")
    args = parser.parse_args()

    for path in (args.video, args.model):
        if not path.is_file():
            parser.error(f"File not found: {path}")
    if args.output.resolve() in (args.video.resolve(), args.model.resolve()):
        parser.error("Choose an output path distinct from the video and model.")

    try:
        frames = extract(args.video, args.model)
        args.output.write_text(
            json.dumps({"frames": frames}, allow_nan=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    except ImportError:
        parser.error("Install the video extras with python -m pip install -r video_tools/requirements.txt.")
    except (OSError, RuntimeError, ValueError) as error:
        parser.error(str(error))

    print(f"Wrote {len(frames)} frames to {args.output}.")


if __name__ == "__main__":
    main()
