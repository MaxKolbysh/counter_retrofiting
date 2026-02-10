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
