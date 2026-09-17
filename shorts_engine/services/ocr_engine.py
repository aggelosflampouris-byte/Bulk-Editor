import cv2
import logging
from pathlib import Path
from PIL import Image

logger = logging.getLogger(__name__)

class OCREngine:
    """
    Extracts text from video frames using Gemini Vision to provide visual context.
    """
    def __init__(self, gemini_api_key: str):
        if not gemini_api_key:
            raise ValueError("Gemini API key is required for OCR extraction")
        self.gemini_api_key = gemini_api_key
        logger.info("Initializing OCREngine with Gemini Vision.")

    def extract_text_from_video(self, video_path: Path, sample_rate_sec: int = 5) -> str:
        """
        Extracts on-screen text from a video by sampling frames and sending them to Gemini.
        
        Args:
            video_path: Path to the video file.
            sample_rate_sec: Extract text from one frame every X seconds.
            
        Returns:
            A consolidated string of all unique text found in the video.
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if fps <= 0:
            logger.warning(f"Could not determine FPS for {video_path}. Defaulting to 30.")
            fps = 30.0
            
        frame_interval = int(fps * sample_rate_sec)
        
        frames = []
        frame_count = 0
        
        logger.info(f"Starting frame extraction on {video_path} (sampling every {sample_rate_sec}s)")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR (OpenCV) to RGB (PIL)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb_frame)
                # Resize to save bandwidth
                pil_img.thumbnail((800, 800))
                frames.append(pil_img)
                
            frame_count += 1
            
        cap.release()
        
        if not frames:
            return ""

        logger.info(f"Sending {len(frames)} frames to Gemini Vision for OCR...")
        
        try:
            from google import genai
            client = genai.Client(api_key=self.gemini_api_key)
            prompt = "Extract all text you see in these images. Return ONLY the text, separated by spaces. Ignore meaningless single letters or numbers. Write the extracted text sequentially."
            
            # Send chunks if there are too many frames (Gemini handles ~15 images easily per prompt)
            max_frames_per_call = 15
            all_text = set()
            
            for i in range(0, len(frames), max_frames_per_call):
                chunk = frames[i:i + max_frames_per_call]
                response = client.models.generate_content(
                    model="gemini-1.5-flash",
                    contents=[prompt] + chunk,
                )
                if response and response.text:
                    all_text.add(response.text.strip())
                    
            consolidated = " | ".join(all_text)
            logger.info("Successfully extracted visual context from video.")
            return consolidated
        except Exception as e:
            logger.error(f"Gemini OCR failed: {e}")
            return ""
