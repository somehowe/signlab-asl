# Convert a recorded sign to landmarks

The optional extractor turns a short video into the same 48-point JSON layout
the SignLab app accepts. It runs locally on the CPU. The output contains
coordinates only.

1. Install the video dependency with Python 3.12.

   ```sh
   python -m pip install -r video_tools/requirements.txt
   ```

   If Ubuntu reports a missing `libGLESv2.so.2`, install the
   [`libgles2` package](https://packages.ubuntu.com/noble/libgles2).

2. Download Google's Holistic Landmarker model and save it as
   `holistic_landmarker.task`. [Official model asset](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task)
3. Trim a video to one supported sign, up to eight seconds. Use a front-facing,
   unmirrored recording with your hands and shoulders visible. A
   constant-frame-rate MP4 is preferred.
4. Run the extractor from the SignLab directory.

   ```sh
   python video_tools/extract_landmarks.py sign.mp4 --model holistic_landmarker.task --output sign.json
   ```

5. Choose **Load landmark JSON** in the app and select `sign.json`.

Each frame contains 21 anatomical left-hand points, 21 anatomical right-hand
points, then the left and right shoulders, elbows, and hips. Missing coordinates
are represented by `null`. Frames are sampled at up to 24 FPS.

The benchmark measures predictions on pre-extracted CLERC Épée landmark
segments. The benchmark numbers do not measure accuracy on new videos
processed by this extractor. SignLab expects a known segment boundary and a
single sign from its supported vocabulary.

[MediaPipe Holistic Landmarker Python API](https://ai.google.dev/edge/api/mediapipe/python/mp/tasks/vision/HolisticLandmarker)
