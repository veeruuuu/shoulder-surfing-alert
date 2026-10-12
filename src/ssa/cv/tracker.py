"""Simple centroid + IoU tracker (§6: "own code, no extra dependency").
§8 contract: tracker.update(raw_faces) -> list[TrackedFace] with stable track_id."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ssa.cv.detector import RawFace


@dataclass
class TrackedFace:
    track_id: int
    bbox: tuple[int, int, int, int]
    landmarks: np.ndarray


def _iou(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = aw * ah
    area_b = bw * bh
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def _centroid(box: tuple[int, int, int, int]) -> tuple[float, float]:
    x, y, w, h = box
    return (x + w / 2.0, y + h / 2.0)


class Tracker:
    """Assigns stable track_ids to faces across frames using IoU, with a
    centroid-distance fallback for fast motion or skipped detection frames."""

    def __init__(self, iou_threshold: float = 0.3, max_centroid_dist: float = 80.0,
                 max_missed_frames: int = 10) -> None:
        self._iou_threshold = iou_threshold
        self._max_centroid_dist = max_centroid_dist
        self._max_missed_frames = max_missed_frames
        self._next_id = 0
        self._tracks: dict[int, TrackedFace] = {}
        self._missed: dict[int, int] = {}

    def update(self, raw_faces: list[RawFace]) -> list[TrackedFace]:
        unmatched_raw = list(range(len(raw_faces)))
        matched_track_ids: set[int] = set()
        results: list[TrackedFace] = []

        for track_id, track in list(self._tracks.items()):
            best_idx = -1
            best_score = 0.0
            for i in unmatched_raw:
                score = _iou(track.bbox, raw_faces[i].bbox)
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1 and best_score >= self._iou_threshold:
                matched = raw_faces[best_idx]
                updated = TrackedFace(track_id=track_id, bbox=matched.bbox, landmarks=matched.landmarks)
                self._tracks[track_id] = updated
                self._missed[track_id] = 0
                matched_track_ids.add(track_id)
                results.append(updated)
                unmatched_raw.remove(best_idx)
            else:
                best_idx = -1
                best_dist = self._max_centroid_dist
                tc = _centroid(track.bbox)
                for i in unmatched_raw:
                    rc = _centroid(raw_faces[i].bbox)
                    dist = float(np.hypot(tc[0] - rc[0], tc[1] - rc[1]))
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = i
                if best_idx != -1:
                    matched = raw_faces[best_idx]
                    updated = TrackedFace(track_id=track_id, bbox=matched.bbox, landmarks=matched.landmarks)
                    self._tracks[track_id] = updated
                    self._missed[track_id] = 0
                    matched_track_ids.add(track_id)
                    results.append(updated)
                    unmatched_raw.remove(best_idx)
                else:
                    self._missed[track_id] = self._missed.get(track_id, 0) + 1

        for i in unmatched_raw:
            track_id = self._next_id
            self._next_id += 1
            new_track = TrackedFace(track_id=track_id, bbox=raw_faces[i].bbox, landmarks=raw_faces[i].landmarks)
            self._tracks[track_id] = new_track
            self._missed[track_id] = 0
            matched_track_ids.add(track_id)
            results.append(new_track)

        for track_id in list(self._tracks.keys()):
            if track_id not in matched_track_ids and self._missed.get(track_id, 0) > self._max_missed_frames:
                del self._tracks[track_id]
                del self._missed[track_id]

        return results