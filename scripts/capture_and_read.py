import time
import os
import json
from datetime import datetime
import sys
import cv2

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
    captured = False
    
    # Try libcamera/rpicam-still (modern Raspberry Pi OS - Bullseye/Bookworm)
    commands = [
        f"rpicam-still -o {IMAGE_PATH} --immediate --nopreview --timeout 1",
        f"libcamera-still -o {IMAGE_PATH} --immediate --nopreview --timeout 1"
    ]
    
    for cmd in commands:
        print(f"Attempting to capture using: {cmd.split()[0]}...")
        ret = os.system(cmd)
        if ret == 0:
            print(f"Image captured successfully using {cmd.split()[0]}.")
            captured = True
            break
    
    if not captured:
        # Try legacy picamera library
        try:
            from picamera import PiCamera
            with PiCamera() as camera:
                camera.resolution = (1024, 768)
                camera.start_preview()
                # Camera warm-up time
                time.sleep(2)
                camera.capture(IMAGE_PATH)
            print("Image captured successfully using legacy Picamera.")
            captured = True
        except (ImportError, Exception):
            pass

    if not captured:
        if os.path.exists(IMAGE_PATH):
            print("Camera not found, but found existing image. Running in simulation mode.")
        else:
            print("Camera not found and no simulation image found.")
            print(f"Please provide an image at {IMAGE_PATH} for simulation.")
            return

    # Process
    try:
        processed = reader.preprocess_image(IMAGE_PATH)
        # Save processed for debugging
        cv2.imwrite(PROCESSED_PATH, processed)
        
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
