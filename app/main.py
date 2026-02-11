from flask import Flask, render_template, jsonify, send_from_directory, request
import os
import json
from datetime import datetime
import sys
import subprocess
import cv2

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.processor import WaterMeterReader

app = Flask(__name__)

# Path to store readings
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_DIR = os.path.join(BASE_DIR, 'data')
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
READINGS_FILE = os.path.join(DATA_DIR, 'readings.json')
CONFIG_FILE = os.path.join(DATA_DIR, 'config.json')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_config():
    if not os.path.exists(CONFIG_FILE):
        return {"rotate": 0, "crop": None}
    with open(CONFIG_FILE, 'r') as f:
        try:
            return json.load(f)
        except:
            return {"rotate": 0, "crop": None}

def get_latest_readings(limit=10):
    if not os.path.exists(READINGS_FILE):
        return []
    with open(READINGS_FILE, 'r') as f:
        try:
            data = json.load(f)
            return data[-limit:]
        except:
            return []

@app.route('/')
def index():
    readings = get_latest_readings()
    latest = readings[-1] if readings else {"timestamp": "N/A", "value": "N/A"}
    return render_template('index.html', latest=latest, history=readings[::-1])

@app.route('/config')
def config():
    readings = get_latest_readings(1)
    latest = readings[0] if readings else {"timestamp": "N/A", "value": "N/A"}
    return render_template('config.html', latest=latest, config=get_config())

@app.route('/api/config', methods=['POST'])
def save_config():
    new_config = request.json
    with open(CONFIG_FILE, 'w') as f:
        json.dump(new_config, f, indent=4)
    return jsonify({"status": "success"})

@app.route('/api/clear_history', methods=['POST'])
def clear_history():
    if os.path.exists(READINGS_FILE):
        with open(READINGS_FILE, 'w') as f:
            json.dump([], f)
    return jsonify({"status": "success"})

@app.route('/api/capture', methods=['POST'])
def capture_now():
    config = get_config()
    rotate = config.get('rotate', 0)
    crop = config.get('crop')
    
    # Capture (Hardware rotation is kept at 0 to match UI manual rotation)
    cmd = f"rpicam-still -o {IMAGES_DIR}/latest.jpg --width 1024 --height 768 --immediate --nopreview --timeout 2000"
    try:
        subprocess.run(cmd, shell=True, check=True)
        
        # Process immediately
        reader = WaterMeterReader()
        processed = reader.preprocess_image(
            os.path.join(IMAGES_DIR, "latest.jpg"), 
            crop=crop, 
            rotate=rotate
        )
        cv2.imwrite(os.path.join(IMAGES_DIR, "processed.jpg"), processed)
        
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/api/latest')
def api_latest():
    readings = get_latest_readings(1)
    return jsonify(readings[0] if readings else {})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
