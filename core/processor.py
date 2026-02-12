import cv2
import numpy as np
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class WaterMeterReader:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            self.client = genai.Client(api_key=api_key)
            self.model_id = "gemini-2.0-flash"
        else:
            self.client = None
            print("WARNING: GEMINI_API_KEY not found in environment.")

    def preprocess_image(self, image_path, crop=None, rotate=0):
        """Simplified Preprocess: Crop original, then rotate the small piece."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Crop from the ORIGINAL unrotated image
        # Cropper.js 'getData(true)' provides coordinates relative to the original file
        if crop and all(k in crop for k in ['x', 'y', 'w', 'h']):
            x, y, w, h = int(crop['x']), int(crop['y']), int(crop['w']), int(crop['h'])
            H_orig, W_orig = img.shape[:2]
            
            # Clamp coordinates
            x = max(0, min(x, W_orig - 1))
            y = max(0, min(y, H_orig - 1))
            w = max(1, min(w, W_orig - x))
            h = max(1, min(h, H_orig - y))
            
            img = img[y:y+h, x:x+w]

        # 2. Rotate ONLY the small cropped piece
        if rotate != 0 and img.size > 0:
            (h, w) = img.shape[:2]
            center = (w / 2, h / 2)
            # OpenCV is CCW, JS is CW. Negate rotate.
            M = cv2.getRotationMatrix2D(center, -rotate, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # 3. Upscale slightly for AI clarity
        if img.shape[0] < 200 and img.size > 0:
            img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        return img

    def read_numbers(self, processed_image):
        """Use modern Google GenAI SDK to extract numbers."""
        if not self.client:
            return "ERR_NO_KEY"
        if processed_image is None or processed_image.size == 0:
            return "ERR_EMPTY_IMG"

        try:
            rgb_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            _, buffer = cv2.imencode('.jpg', rgb_img)
            
            prompt = "Read the numeric value from this water meter counter. Return ONLY the digits as a single number. This is a mechanical counter; some digits might be halfway turned."
            
            response = self.client.models.generate_content(
                model=self.model_id,
                contents=[
                    prompt,
                    types.Part.from_bytes(data=buffer.tobytes(), mime_type="image/jpeg")
                ]
            )

            result = response.text.strip()
            digits = "".join(filter(str.isdigit, result))
            print(f"Gemini Response: {result} -> Extracted: {digits}")
            return digits

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "ERROR"
