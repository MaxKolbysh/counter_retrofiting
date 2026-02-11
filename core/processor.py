import cv2
import numpy as np
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class WaterMeterReader:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash')
        else:
            self.model = None
            print("WARNING: GEMINI_API_KEY not found in environment.")

    def preprocess_image(self, image_path, crop=None, rotate=0):
        """Preprocess the image: Rotate and Crop."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Crop FIRST (Relative to original image)
        if crop and all(k in crop for k in ['x', 'y', 'w', 'h']):
            x, y, w, h = int(crop['x']), int(crop['y']), int(crop['w']), int(crop['h'])
            H_orig, W_orig = img.shape[:2]
            x = max(0, min(x, W_orig - 1))
            y = max(0, min(y, H_orig - 1))
            w = max(1, min(w, W_orig - x))
            h = max(1, min(h, H_orig - y))
            img = img[y:y+h, x:x+w]

        # 2. Rotate the cropped result
        if rotate != 0:
            (h, w) = img.shape[:2]
            angle = -rotate  # Match JS/CSS rotation direction
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        return img

    def read_numbers(self, processed_image):
        """Use Gemini Vision to extract numbers from the image."""
        if not self.model:
            return "ERR_NO_KEY"

        try:
            # Convert OpenCV image (BGR) to RGB for Gemini
            rgb_img = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Encode image to JPEG buffer
            _, buffer = cv2.imencode('.jpg', rgb_img)
            image_data = buffer.tobytes()

            # Prepare the prompt
            prompt = "Read the numeric value from this water meter counter. Return ONLY the digits as a single number. If you are unsure or some digits are partially turned, provide your best guess of the digits visible."

            # Call Gemini
            response = self.model.generate_content([
                prompt,
                {'mime_type': 'image/jpeg', 'data': image_data}
            ])

            # Clean up the response (remove any non-digits like spaces or decimals if Gemini adds them)
            result = response.text.strip()
            # Extract only digits
            digits = "".join(filter(str.isdigit, result))
            
            print(f"Gemini Response: {result} -> Extracted: {digits}")
            return digits

        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return f"ERROR"

if __name__ == "__main__":
    reader = WaterMeterReader()
    print("Reader initialized with Gemini support.")
