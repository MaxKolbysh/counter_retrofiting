import cv2
import pytesseract
import numpy as np
import os

class WaterMeterReader:
    def __init__(self, tesseract_cmd=None, templates_dir=None):
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.templates_dir = templates_dir
        
        # Check if tesseract is installed
        try:
            pytesseract.get_tesseract_version()
        except pytesseract.TesseractNotFoundError:
            print("WARNING: Tesseract not found. Falling back to OpenCV only.")

    def preprocess_image(self, image_path, crop=None, rotate=0):
        """Preprocess the image to highlight numbers."""
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")

        # 1. Crop FIRST
        if crop and all(k in crop for k in ['x', 'y', 'w', 'h']):
            x, y, w, h = int(crop['x']), int(crop['y']), int(crop['w']), int(crop['h'])
            H_orig, W_orig = img.shape[:2]
            x = max(0, min(x, W_orig - 1))
            y = max(0, min(y, H_orig - 1))
            w = max(1, min(w, W_orig - x))
            h = max(1, min(h, H_orig - y))
            img = img[y:y+h, x:x+w]

        # 2. Rotate
        if rotate != 0:
            (h, w) = img.shape[:2]
            angle = -rotate
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(0,0,0))

        # 3. Grayscale & Upscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

        # 4. Adaptive Thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # 5. Noise removal
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        return processed

    def segment_digits(self, processed_image, num_digits=5):
        """Split the cropped image into individual digit boxes."""
        h, w = processed_image.shape[:2]
        digit_w = w // num_digits
        digits = []
        for i in range(num_digits):
            x = i * digit_w
            digit_img = processed_image[0:h, x:x+digit_w]
            digits.append(digit_img)
        return digits

    def match_digit(self, digit_img, templates):
        """Find the best matching template for a digit image."""
        best_match = None
        max_val = -1
        
        for label, template in templates.items():
            # Resize template to match digit_img size if necessary
            if template.shape != digit_img.shape:
                template = cv2.resize(template, (digit_img.shape[1], digit_img.shape[0]))
            
            res = cv2.matchTemplate(digit_img, template, cv2.TM_CCOEFF_NORMED)
            min_val, val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if val > max_val:
                max_val = val
                best_match = label
        
        return best_match if max_val > 0.6 else "?"

    def read_numbers(self, processed_image, num_digits=5):
        """Extract numbers using a mix of OCR and OpenCV template matching."""
        # Try Tesseract first
        custom_config = r'--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789'
        text = pytesseract.image_to_string(processed_image, config=custom_config).strip()
        
        # If Tesseract fails or is too short, try Template Matching if we have templates
        if (not text or len(text) < num_digits) and self.templates_dir and os.path.exists(self.templates_dir):
            print("Tesseract failed or incomplete. Trying template matching...")
            # Load templates (this should ideally be done once in __init__)
            # For simplicity, we assume templates are named 0.jpg, 1.jpg ...
            templates = {}
            for i in range(10):
                path = os.path.join(self.templates_dir, f"{i}.jpg")
                if os.path.exists(path):
                    templates[str(i)] = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            
            if templates:
                digit_imgs = self.segment_digits(processed_image, num_digits=num_digits)
                matched_text = "".join([self.match_digit(d, templates) for d in digit_imgs])
                return matched_text
                
        return text

if __name__ == "__main__":
    reader = WaterMeterReader()
    print("Reader initialized.")
