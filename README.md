# Bharat AI-SoC Project: Road Anomaly Detection

## Project Overview
This project detects road anomalies such as **potholes, bumps, and cracks** in videos using a YOLO-based **ONNX model**.  
The system is deployed on a **Raspberry Pi 4** and can process pre-recorded videos to highlight anomalies with bounding boxes, confidence scores, and FPS information.  

The project demonstrates a complete end-to-end pipeline:
- Preprocessing video frames
- Running inference on a YOLO model
- Post-processing with class-specific thresholds and pothole filtering
- Annotating frames with bounding boxes, labels, timestamp, and FPS
- Generating output videos showing detected anomalies

## Instructions and Project Features and Dependencies 

### 1. Run the default test video:

```bash
python3 main.py

### Dependencies

1.Python 3.x

2.OpenCV (opencv-python)

3.NumPy (numpy)

4.ONNX Runtime (onnxruntime)

### Install required packages via pip:

pip install -r requirements.txt

### Project Features:

* YOLO-based ONNX model for road anomaly detection

* Modular code for preprocessing, inference, and post-processing

* Class-specific confidence thresholds and filtering for potholes

* Bounding boxes, confidence labels, timestamp, and FPS overlay on output video

* Real-time demo captured from Raspberry Pi 4

* Fully relative paths for easy repository cloning 

### Author's

Paul Rojer R 
Brindha S
Sriram A
