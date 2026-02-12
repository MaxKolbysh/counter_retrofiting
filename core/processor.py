import cv2
import numpy as np
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class WaterMeterReader:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-2.0-flash"
        else:
            self.client = None

    def preprocess_image(self, image_path, crop=None, rotate=0):
        """
        Processes image to match Cropper.js behavior:
        1. Rotate full image around its center (1024x768).
        2. Crop using coordinates relative to that rotated canvas.
        """
        img = cv2.imread(image_path)
        if img is None:
            return None

        # Ensure we are working with the expected base resolution
        # if the camera captured something else
        if img.shape[0] != 768 or img.shape[1] != 1024:
            img = cv2.resize(img, (1024, 768))

        # 1. Rotate the full 1024x768 image
        # We keep the size 1024x768 to match how coordinates are reported
        if rotate != 0:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            # OpenCV rotation is CCW, JS is CW. Negate the angle.
            M = cv2.getRotationMatrix2D(center, -rotate, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # 2. Crop from the rotated image
        if crop and all(k in crop for k in ['x', 'y', 'w', 'h']):
            x, y, cw, ch = int(crop['x']), int(crop['y']), int(crop['w']), int(crop['h'])
            H, W = img.shape[:2]
            
            # Boundary clamping
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            cw = max(1, min(cw, W - x))
            ch = max(1, min(ch, H - y))
            
            img = img[y:y+ch, x:x+cw]

        return img

    def read_numbers(self, processed_image):
        if not self.client or processed_image is None or processed_image.size == 0:
            return "N/A"

        try:
            rgb_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rgb_img)
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    "Read the water meter digits. Return only the numbers.",
                    types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
                ]
            )
            val = "".join(filter(str.isdigit, response.text.strip()))
            print(f"Gemini Read: {val}")
            return val
        except Exception as e:
            print(f"Gemini Error: {e}")
            return "ERROR"
