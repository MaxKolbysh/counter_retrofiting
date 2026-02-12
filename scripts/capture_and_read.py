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
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    IMAGES_DIR = os.path.join(DATA_DIR, 'images')
    IMAGE_PATH = os.path.join(IMAGES_DIR, 'latest.jpg')
    PROCESSED_PATH = os.path.join(IMAGES_DIR, 'processed.jpg')
    READINGS_FILE = os.path.join(DATA_DIR, 'readings.json')
    CONFIG_PATH = os.path.join(DATA_DIR, 'config.json')
    
    os.makedirs(IMAGES_DIR, exist_ok=True)
    reader = WaterMeterReader()
    
    # Reload config every time
    rotate = 0
    crop = None
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                cfg = json.load(f)
                rotate = cfg.get('rotate', 0)
                crop = cfg.get('crop')
                print(f"Loaded config: rotate={rotate}, crop={crop}")
        except Exception as e:
            print(f"Config load error: {e}")

    # Capture
    cmd = f"rpicam-still -o {IMAGE_PATH} --width 1024 --height 768 --immediate --nopreview --timeout 2000"
    captured = False
    try:
        ret = subprocess.run(cmd, shell=True, timeout=15)
        if ret.returncode == 0:
            captured = True
    except:
        pass
    
    if not captured and os.path.exists(IMAGE_PATH):
        captured = True

    if captured:
        try:
            processed = reader.preprocess_image(IMAGE_PATH, crop=crop, rotate=rotate)
            if processed is not None:
                cv2.imwrite(PROCESSED_PATH, processed)
                value = reader.read_numbers(processed)
                
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] Value: {value}")
                
                if value and value not in ["ERROR", "N/A"]:
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
            print(f"Process error: {e}")

if __name__ == "__main__":
    print("Background capture service running...")
    while True:
        main()
        time.sleep(60)
