import cv2
import pytesseract
import numpy as np

class WaterMeterReader:
    def __init__(self, tesseract_cmd=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        
        # Check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("WARNING: Tesseract not found. OCR will fail.")
            print("Install it with: sudo apt install tesseract-ocr")

    def preprocess_image(self, image_path, crop=None, rotate=0):
        """Preprocess the image to highlight numbers."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Crop FIRST
        # Cropper.js 'data' (x, y, w, h) are relative to the original unrotated image.
        if crop and all(k in crop for k in ['x', 'y', 'w', 'h']):
            x, y, w, h = int(crop['x']), int(crop['y']), int(crop['w']), int(crop['h'])
            H_orig, W_orig = img.shape[:2]
            
            # Clamp to prevent out-of-bounds errors
            x = max(0, min(x, W_orig - 1))
            y = max(0, min(y, H_orig - 1))
            w = max(1, min(w, W_orig - x))
            h = max(1, min(h, H_orig - y))
            
            img = img[y:y+h, x:x+w]
            if img.size == 0:
                raise ValueError("Invalid crop coordinates resulted in empty image")

        # 2. Rotate the cropped result
        if rotate != 0:
            (h, w) = img.shape[:2]
            # OpenCV is CCW, JS is CW. Negate to match.
            angle = -rotate
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # 3. Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 4. Upscale image (Tesseract likes larger text)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 5. Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # 6. Noise removal
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return processed

    def read_numbers(self, processed_image):
        """Use Tesseract to extract numbers from the processed image."""
        # PSM 6: Assume a single uniform block of text.
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        
        # Pytesseract expects RGB or PIL image
        text = pytesseract.image_to_string(processed_image, config=custom_config)
        print(f"DEBUG: OCR raw output: '{text}'")
        return text.strip()

if __name__ == "__main__":
    # Test block
    reader = WaterMeterReader()
    print("Reader initialized.")
