import cv2
import numpy as np

def preprocess(frame, input_size=320):
    """Resize and normalize frame for ONNX input."""
    img = cv2.resize(frame, (input_size, input_size))
    img = img / 255.0
    img = img.transpose(2, 0, 1)
    img = np.expand_dims(img, axis=0).astype(np.float32)
    return img