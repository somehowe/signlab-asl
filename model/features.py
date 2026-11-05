"""Turn a clip of landmark coordinates into fixed-size classifier inputs."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Layout:
    points: int
    dimensions: int
    hand_indices: tuple[tuple[int, ...], ...] = ()
    shoulder_indices: tuple[int, ...] = ()
    steps: int = 12

    def __post_init__(self):
        if self.points < 3 or self.dimensions not in (2, 3) or self.steps < 2:
            raise ValueError("Expected at least 3 landmarks, 2 or 3 coordinates, and 2 time steps.")
        if self.shoulder_indices and len(self.shoulder_indices) != 2:
            raise ValueError("Shoulder metadata must contain two landmark indices.")
        for hand in self.hand_indices:
            if len(hand) != 21 or len(set(hand)) != 21:
                raise ValueError("Each hand must contain 21 unique MediaPipe landmark indices.")
        indices = (*self.shoulder_indices, *(point for hand in self.hand_indices for point in hand))
        if any(point < 0 or point >= self.points for point in indices):
            raise ValueError("A landmark index is outside the configured layout.")


class LowSignal(ValueError):
    """The clip lacks enough visible landmarks for a useful prediction."""


def _body_origin_scale(frames, visible, layout):
    if layout.shoulder_indices:
        left, right = layout.shoulder_indices
        paired = visible[:, left] & visible[:, right]
        if np.any(paired):
            shoulders = frames[paired][:, [left, right]]
            widths = np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1)
            if np.median(widths) > 1e-8:
                return np.median(shoulders.mean(axis=1), axis=0), float(np.median(widths))

    points = frames[visible]
    origin = np.median(points, axis=0)
    spans = []
    for hand in layout.hand_indices:
        wrist, middle_knuckle = hand[0], hand[9]
        paired = visible[:, wrist] & visible[:, middle_knuckle]
        spans.extend(np.linalg.norm(
            frames[paired, wrist, :2] - frames[paired, middle_knuckle, :2], axis=1
        ))
    if spans and np.median(spans) > 1e-8:
        return origin, float(np.median(spans))
    scale = float(np.quantile(np.linalg.norm(points[:, :2] - origin[:2], axis=1), 0.75))
    if scale <= 1e-8:
        raise LowSignal("The clip does not show spatially distinct landmarks.")
    return origin, scale


def feature_pair(frames, layout):
    """Return mean-pose and time-ordered features using the same normalized clip."""
    frames = np.asarray(frames, dtype=np.float64)
    if frames.ndim != 3 or frames.shape[1:] != (layout.points, layout.dimensions):
        raise ValueError(
            f"Expected frames with shape [time, {layout.points}, {layout.dimensions}]."
        )
    visible = np.isfinite(frames).all(axis=2)
    if not len(frames) or visible.sum(axis=1).max() < 3:
        raise LowSignal("The clip has fewer than three visible landmarks in every frame.")
    if layout.hand_indices and not any(
        visible[:, hand].sum(axis=1).max() >= 5 for hand in layout.hand_indices
    ):
        raise LowSignal("The clip does not contain enough visible hand landmarks.")

    origin, scale = _body_origin_scale(frames, visible, layout)
    points = np.clip((frames - origin) / scale, -8, 8)
    channels = [points]
    masks = [visible]
    for hand in layout.hand_indices:
        wrist = hand[0]
        channels.append(np.clip((frames[:, hand] - frames[:, wrist, None]) / scale, -8, 8))
        masks.append(visible[:, hand] & visible[:, wrist, None])
    points = np.concatenate(channels, axis=1)
    visible = np.concatenate(masks, axis=1)

    times = np.linspace(0, len(frames) - 1, layout.steps)
    timeline = np.arange(len(frames))
    sampled = np.zeros((layout.steps, points.shape[1], layout.dimensions))
    for index in np.flatnonzero(visible.any(axis=0)):
        observed = visible[:, index]
        for axis in range(layout.dimensions):
            sampled[:, index, axis] = np.interp(
                times, timeline[observed], points[observed, index, axis]
            )
    presence = visible.mean(axis=0)
    pooled = np.concatenate((sampled.mean(axis=0).ravel(), presence))
    temporal = np.concatenate((pooled, sampled.ravel()))
    return pooled, temporal
