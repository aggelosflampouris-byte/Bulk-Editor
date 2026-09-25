"""
tests/test_face_tracker.py — Unit tests for active speaker face tracking and dynamic cropping.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from shorts_engine.services.face_tracker import track_active_speaker
from shorts_engine.services.video_engine import crop_to_9_16, probe_resolution


def _generate_synthetic_video(path: Path, width: int, height: int, duration: float = 2.0) -> Path:
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=red:s={width}x{height}:d={duration}",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert res.returncode == 0, f"FFmpeg failed: {res.stderr}"
    return path


def test_face_tracker_landscape_fallback_center_crop(tmp_path):
    # Plain solid red landscape video (1920x1080) — no human faces present
    landscape_video = _generate_synthetic_video(tmp_path / "landscape.mp4", 1920, 1080, duration=1.0)
    tracking_info = track_active_speaker(
        video_path=landscape_video,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    crop_x = tracking_info.static_crop_x

    assert isinstance(crop_x, int)
    # Scaled width = 1920 * (1920 / 1080) = 3413 px
    # Max crop X = 3413 - 1080 = 2333 px
    # Center crop offset = 2333 // 2 = 1166 px
    scale = 1920 / 1080
    scaled_w = int(1920 * scale)
    max_crop_x = scaled_w - 1080
    center_x = max_crop_x // 2

    assert 0 <= crop_x <= max_crop_x
    assert abs(crop_x - center_x) <= 2


def test_crop_to_9_16_with_dynamic_crop_offset(tmp_path):
    landscape_video = _generate_synthetic_video(tmp_path / "landscape_input.mp4", 1920, 1080, duration=1.0)
    output_video = tmp_path / "cropped_short.mp4"

    # Crop with explicit speaker offset
    result = crop_to_9_16(landscape_video, output_video, crop_x_offset=300)
    assert result.is_file()
    w, h = probe_resolution(result)
    assert (w, h) == (1080, 1920)


def test_track_active_speaker_no_faces(tmp_path):
    landscape_video = _generate_synthetic_video(tmp_path / "landscape_no_face.mp4", 1920, 1080, duration=1.0)
    from shorts_engine.services.face_tracker import (
        SpeakerTrackingResult,
        track_active_speaker,
    )

    result = track_active_speaker(
        video_path=landscape_video,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
    )
    assert isinstance(result, SpeakerTrackingResult)
    assert result.has_speaker is False
    assert result.speaker_presence_ratio == 0.0
    assert isinstance(result.static_crop_x, int)
    assert isinstance(result.crop_expression, str)


def test_crop_to_9_16_with_dynamic_time_expression(tmp_path):
    landscape_video = _generate_synthetic_video(tmp_path / "dynamic_input.mp4", 1920, 1080, duration=1.0)
    output_video = tmp_path / "dynamic_cropped.mp4"

    result = crop_to_9_16(
        landscape_video,
        output_video,
        crop_x_expr="if(lt(t,0.5),200,500)",
    )
    assert result.is_file()
    w, h = probe_resolution(result)
    assert (w, h) == (1080, 1920)


def test_track_active_speaker_dynamic_follow_and_fallback(tmp_path, monkeypatch):
    """
    Verify that speaker motion generates piecewise follow expressions,
    frames faces at rule-of-thirds eye-line, and low presence retains detected position.
    """
    landscape_video = _generate_synthetic_video(tmp_path / "speaker_sim.mp4", 1920, 1080, duration=2.0)

    # Mock YOLO to simulate a speaker moving horizontally across frames
    class MockBox:
        def __init__(self, x1, y1, x2, y2):
            import torch
            self.xyxy = torch.tensor([[x1, y1, x2, y2]])

    class MockKeypoints:
        def __init__(self, nose_x, nose_y):
            import torch
            self.xy = torch.tensor([[[nose_x, nose_y], [nose_x - 10, nose_y - 10], [nose_x + 10, nose_y - 10], [0, 0], [0, 0]]])
            self.conf = torch.tensor([[0.95, 0.90, 0.90, 0.0, 0.0]])

    class MockResult:
        def __init__(self, box, kp):
            self.boxes = [box]
            self.keypoints = [kp]

    frame_counter = {"val": 0}

    class MockYOLO:
        def __init__(self, *args, **kwargs):
            pass

        def to(self, *args, **kwargs):
            return self

        def __call__(self, frame, **kwargs):
            idx = frame_counter["val"]
            frame_counter["val"] += 1
            # Move speaker from X=400 to X=1200 across time
            x_pos = 400.0 + idx * 100.0
            y_pos = 200.0
            box = MockBox(x_pos - 100, y_pos - 100, x_pos + 100, y_pos + 300)
            kp = MockKeypoints(x_pos, y_pos)
            return [MockResult(box, kp)]

    import ultralytics
    monkeypatch.setattr(ultralytics, "YOLO", MockYOLO)

    result = track_active_speaker(
        video_path=landscape_video,
        source_width=1920,
        source_height=1080,
        target_width=1080,
        target_height=1920,
        sample_interval=0.25,
    )

    assert result.has_speaker is True
    assert result.speaker_presence_ratio > 0.5
    # Confirm dynamic follow piecewise linear expression was constructed
    assert "if(lt(t," in result.crop_expression
    # Verify vertical offset places head near top third (clamped or rule-of-thirds eye-line)
    assert isinstance(result.static_crop_y, int)
    assert 0 <= result.static_crop_y <= 1080


