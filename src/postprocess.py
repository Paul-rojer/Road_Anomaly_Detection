import numpy as np

CLASS_THRESHOLDS = {0:0.25, 1:0.25, 2:0.30, 3:0.30, 4:0.30}

def filter_boxes(predictions, width, height, input_size=320, conf_threshold=0.25):
    """Filter boxes based on class thresholds and pothole rules."""
    boxes, confidences, class_ids = [], [], []

    for pred in predictions.T:
        scores = pred[4:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if class_id == 1:  # Ignore vehicle class
            continue

        threshold = CLASS_THRESHOLDS.get(class_id, conf_threshold)
        if confidence < threshold:
            continue

        cx, cy, w, h = pred[0:4]
        x1 = int((cx - w/2) * width / input_size)
        y1 = int((cy - h/2) * height / input_size)
        w = int(w * width / input_size)
        h = int(h * height / input_size)

        area = w * h
        if class_id == 0 and (area > 18000 or (h/w > 1.5)):
            continue

        boxes.append([x1, y1, w, h])
        confidences.append(float(confidence))
        class_ids.append(class_id)

    return boxes, confidences, class_ids
