import time
import os
import json
from datetime import datetime
import sys
import cv2
import subprocess
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.processor import WaterMeterReader

# Load .env
load_dotenv()

def main():
    # Paths
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    IMAGE_PATH = os.path.join(IMAGES_DIR, 'latest.jpg')
    PROCESSED_PATH = os.path.join(IMAGES_DIR, 'processed.jpg')
    READINGS_FILE = os.path.join(DATA_DIR, 'readings.json')
    CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')
    
    # Ensure directories exist
    os.makedirs(IMAGES_DIR, exist_ok=True)
    
    # Initialize reader
    reader = WaterMeterReader()
    
    # Load latest config
    rotate = 0
    crop = None
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
                rotate = cfg.get('rotate', 0)
                crop = cfg.get('crop')
        except:
            pass

    # Camera Capture
    captured = False
    cmd = f"rpicam-still -o {IMAGE_PATH} --width 1024 --height 768 --immediate --nopreview --timeout 2000"
    
    try:
        ret = subprocess.run(cmd, shell=True, timeout=15)
        if ret.returncode == 0:
            captured = True
    except:
        pass
    
    # Fallback to simulation if image exists but capture failed
    if not captured and os.path.exists(IMAGE_PATH):
        captured = True

    if captured:
        try:
            # Process using the SAME logic as manual capture (Rotate then Crop)
            processed = reader.preprocess_image(IMAGE_PATH, crop=crop, rotate=rotate)
            
            # Save the processed image so the UI can see it
            cv2.imwrite(PROCESSED_PATH, processed)
            
            # Read with Gemini
            value = reader.read_numbers(processed)
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Read value: '{value}'")
            
            if value and value not in ["ERROR", "ERR_NO_KEY", "ERR_EMPTY_IMG"]:
                # Save reading
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                reading = {"timestamp": timestamp, "value": value}
                
                data = []
                if os.path.exists(READINGS_FILE):
                    with open(READINGS_FILE, 'r') as f:
                        try: data = json.load(f)
                        except: data = []
                
                data.append(reading)
                with open(READINGS_FILE, 'w') as f:
                    json.dump(data[-100:], f, indent=4)
                    
        except Exception as e:
            print(f"Error during processing: {e}")
    else:
        print("Capture failed and no simulation image found.")

if __name__ == "__main__":
    print("Background capture service started...")
    while True:
        main()
        time.sleep(60)
