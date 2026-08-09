# Third-party notices

WALP includes and depends on the following third-party components. They are
owned by their respective authors and licensed under their own terms, which
apply to those components rather than the WALP licence.

---

## Included in this repository

### MediaPipe Pose Landmarker models

- Files: `assets/pose_landmarker_lite.task`, `assets/pose_landmarker_full.task`
- Copyright © Google LLC
- Licence: Apache License, Version 2.0
- <https://developers.google.com/mediapipe/solutions/vision/pose_landmarker>

These model files are redistributed here unmodified so that the game runs
straight from a clone. The Apache 2.0 licence permits redistribution provided
this notice is included.

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use these files except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

---

## Required at runtime, installed separately with pip

These are **not** redistributed here. Each is installed by the user from PyPI
and remains under its own licence.

| Package | Licence | Project |
|---|---|---|
| `mediapipe` | Apache License 2.0 | <https://github.com/google-ai-edge/mediapipe> |
| `opencv-python` | Apache License 2.0 | <https://github.com/opencv/opencv-python> |
| `pygame` | GNU LGPL v2.1 | <https://github.com/pygame/pygame> |
| `numpy` | BSD 3-Clause | <https://github.com/numpy/numpy> |

pygame is used as an unmodified library through its public interface, which
the LGPL permits without imposing its terms on this project.

---

## Not third party

Everything else in this repository — all Python source, and the audio in
`assets/music/` — is the original work of the copyright holder and is covered
by [LICENCE.md](LICENCE.md).
