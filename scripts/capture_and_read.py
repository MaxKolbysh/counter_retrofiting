import time
import os
import json
from datetime import datetime
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.processor import WaterMeterReader

def main():
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    IMAGE_PATH = os.path.join(BASE_DIR, 'data/images/latest.jpg')
    PROCESSED_PATH = os.path.join(BASE_DIR, 'data/images/processed.jpg')
    READINGS_FILE = os.path.join(BASE_DIR, 'data/readings.json')
    
    # Ensure directories exist
    os.makedirs(os.path.dirname(IMAGE_PATH), exist_ok=True)
    
    # Initialize reader
    reader = WaterMeterReader()
    
    # Camera Capture logic
    try:
        from picamera import PiCamera # Legacy or use Picamera2
        # For Camera 2 / Bullseye+ use Picamera2:
        # from picamera2 import Picamera2
        # picam2 = Picamera2()
        # picam2.start()
        # picam2.capture_file(IMAGE_PATH)
        print("Camera detected. Capturing...")
        # (Placeholder for actual capture command)
        # Using shell command as fallback if library is tricky
        os.system(f"libcamera-still -o {IMAGE_PATH} --immediate")
    except ImportError:
        print("Picamera not found. Running in simulation mode.")
        if not os.path.exists(IMAGE_PATH):
            print(f"Please provide an image at {IMAGE_PATH} for simulation.")
            return

    # Process
    try:
        processed = reader.preprocess_image(IMAGE_PATH)
        # Save processed for debugging
        import cv2
        cv2.imwrite('data/images/processed.jpg', processed)
        
        value = reader.read_numbers(processed)
        print(f"Read value: {value}")
        
        if value:
            # Save reading
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            reading = {"timestamp": timestamp, "value": value}
            
            data = []
            if os.path.exists(READINGS_FILE):
                with open(READINGS_FILE, 'r') as f:
                    try:
                        data = json.load(f)
                    except:
                        data = []
            
            data.append(reading)
            with open(READINGS_FILE, 'w') as f:
                json.dump(data, f, indent=4)
                
    except Exception as e:
        print(f"Error during processing: {e}")

if __name__ == "__main__":
    while True:
        main()
        print("Sleeping for 60 seconds...")
        time.sleep(60)
