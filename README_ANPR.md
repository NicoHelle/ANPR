
# Automatic Number Plate Recognition (ANPR)

This project implements a **real-time Automatic Number Plate Recognition (ANPR)** system using Python, Flask, machine learning (YOLO), and OCR. It supports video stream processing and includes super-resolution for enhancing license plate images.

---

## 🧠 Key Features

- **Real-time License Plate Detection** using YOLOv8
- **Optical Character Recognition (OCR)** via EasyOCR
- **Super-resolution enhancement** with RRDBNet (Residual-in-Residual Dense Blocks)
- **Custom web interface** built with Flask and Bootstrap
- **Live video stream processing** from webcam or video file
- **Adjustable detection threshold** via a user-friendly slider in the browser

---

## 📂 Project Structure

- `app.py` — Flask server to handle video stream and routes
- `videoStream.py` — Contains the detection, OCR and super-resolution logic
- `RRDBNet_arch.py` — Neural network architecture for super-resolution
- `index.html` — Frontend with real-time video feed and controls
- `utils/` — (not included here) Contains pre-trained models and resources

---

## 🛠️ Requirements

- Python 3.8+
- Flask
- OpenCV
- torch (PyTorch)
- ultralytics (YOLOv8)
- easyocr
- matplotlib
- numpy
- Pillow

Install dependencies using:

```bash
pip install -r requirements.txt
```

Or manually install:

```bash
pip install flask opencv-python torch torchvision easyocr ultralytics matplotlib numpy pillow
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ANPR.git
cd ANPR
```

2. Make sure you have the required model files in `utils/`, including:
   - `best.pt` (YOLOv8 trained weights)
   - `RRDB_PSNR_x4.pth` (super-resolution model)
   - `ESPCN_x4.pb` (OpenCV DNN super-res model)

3. Start the Flask app:
```bash
python app.py
```

4. Open your browser and navigate to:
```
http://localhost:5000
```

---

## 🧪 Example Workflow

- The system processes each video frame using YOLO to detect plates.
- Detected regions are enhanced using super-resolution (RRDBNet).
- EasyOCR reads the plate text from processed regions.
- Results are displayed on the video stream in real-time.

---

## 📄 License

MIT License

---

## 👨‍💻 Author

**Nico Helle**  
Based on a university project for real-time license plate detection and OCR.

---
