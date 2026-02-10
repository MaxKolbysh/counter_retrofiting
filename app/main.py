from flask import Flask, render_template, jsonify, send_from_directory
import os
import json
from datetime import datetime

app = Flask(__name__)

# Path to store readings
DATA_DIR = 'data'
IMAGES_DIR = os.path.join(DATA_DIR, 'images')
READINGS_FILE = os.path.join(DATA_DIR, 'readings.json')

# Ensure directories exist
os.makedirs(IMAGES_DIR, exist_ok=True)

def get_latest_readings(limit=10):
    if not os.path.exists(READINGS_FILE):
        return []
    with open(READINGS_FILE, 'r') as f:
        try:
            data = json.load(f)
            return data[-limit:]
        except json.JSONDecodeError:
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
    return render_template('config.html', latest=latest)

@app.route('/api/latest')
def api_latest():
    readings = get_latest_readings(1)
    return jsonify(readings[0] if readings else {})

@app.route('/images/<filename>')
def serve_image(filename):
    return send_from_directory(IMAGES_DIR, filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
