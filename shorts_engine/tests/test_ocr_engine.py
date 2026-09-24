from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from shorts_engine.services.ocr_engine import OCREngine


@pytest.fixture
def mock_genai():
    with patch("google.genai.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_response = MagicMock()
        mock_response.text = "Mocked visual text from Gemini"
        mock_client.models.generate_content.return_value = mock_response
        yield mock_client_cls, mock_client

@pytest.fixture
def mock_cv2():
    with patch("shorts_engine.services.ocr_engine.cv2") as mock_cv2_module:
        mock_cap = MagicMock()
        mock_cv2_module.VideoCapture.return_value = mock_cap
        
        # Mock frame properties
        mock_cv2_module.CAP_PROP_FPS = 5
        mock_cv2_module.COLOR_BGR2RGB = 4
        
        import numpy as np
        # Mock a dummy rgb frame to pass to PIL
        mock_cv2_module.cvtColor.return_value = np.zeros((10, 10, 3), dtype=np.uint8)
        
        yield mock_cv2_module, mock_cap

def test_ocr_engine_initialization():
    engine = OCREngine(gemini_api_key="test_key")
    assert engine.gemini_api_key == "test_key"

def test_ocr_engine_missing_key():
    with pytest.raises(ValueError, match="Gemini API key is required"):
        OCREngine(gemini_api_key="")

def test_extract_text_from_video(mock_genai, mock_cv2, tmp_path):
    _mock_genai_module, mock_client = mock_genai
    _mock_cv2_module, mock_cap = mock_cv2
    
    # Create a dummy video file so path.exists() is true
    dummy_vid = tmp_path / "dummy.mp4"
    dummy_vid.touch()
    
    # Mock FPS and VideoCapture behavior
    mock_cap.get.return_value = 30.0
    mock_cap.isOpened.return_value = True
    
    # We want cap.read() to return True, frame for 3 frames, then False
    dummy_frame = object()
    mock_cap.read.side_effect = [
        (True, dummy_frame),  # Frame 0 (processed)
        (True, dummy_frame),  # Frame 1 (skipped)
        (True, dummy_frame),  # Frame 2 (skipped)
        (False, None)         # End
    ]
    
    engine = OCREngine(gemini_api_key="test_key")
    # sample_rate_sec=1 with 30fps means it checks every 30 frames.
    # Since we only mocked 3 frames, it will only process Frame 0.
    result = engine.extract_text_from_video(dummy_vid, sample_rate_sec=1)
    
    assert "Mocked visual text from Gemini" in result
    mock_client.models.generate_content.assert_called_once()

def test_extract_text_file_not_found():
    engine = OCREngine(gemini_api_key="test_key")
    with pytest.raises(FileNotFoundError):
        engine.extract_text_from_video(Path("does_not_exist.mp4"))
