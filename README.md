# Water Meter Retrofit (Pi Zero 2 + Camera 2)

Project to read mechanical water meter numbers using OCR.

## Setup on Raspberry Pi

1. **Install System Dependencies:**
   ```bash
   sudo apt update
   sudo apt install -y tesseract-ocr libcamera-apps python3-opencv
   ```

2. **Clone/Transfer Project:**
   Use SCP or Git to move this folder to your Pi.

3. **Install Python Dependencies:**
   It is recommended to use `uv` for fast dependency management.
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   uv sync
   ```

## Running the Project

1. **Start the Web Dashboard:**
   ```bash
   uv run python app/main.py
   ```
   Access it at `http://<pi-ip-address>:5000`

2. **Start the Capture & OCR Service:**
   ```bash
   uv run python scripts/capture_and_read.py
   ```

## Tuning OCR
The image processing logic in `core/processor.py` may need adjustment based on your lighting and meter type. You can adjust the `adaptiveThreshold` parameters or add cropping to focus strictly on the numbers.


to add:

  What's New:
   1. Toggle Switch: A new switch in the Config page allows you to choose between "Cloud Processing (Gemini)" and "Local processing
      (TFLite)".
   2. Persistent Choice: Your choice is saved in config.json.
   3. UI Feedback: The label updates instantly to show which mode is active.


  Next Step for you:
  I haven't written the Local Python logic yet because we don't have the .tflite model file. Once you provide the model, I can update
  core/processor.py to use it when the toggle is off.


  For now, the system is ready for the future! If you toggle it to "Local", it will still run the capture, but the Python script will
  need the TFLite logic to actually read the numbers.


  To apply: git pull origin master and restart the app. Check out the new switch!

  