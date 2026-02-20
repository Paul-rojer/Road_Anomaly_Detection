import cv2

def draw_boxes(frame, boxes, class_ids, confidences):
    """Draw bounding boxes and labels on the frame."""
    if len(boxes) == 0:
        return frame

    for i in range(len(boxes)):
        x, y, w, h = boxes[i]
        cls = class_ids[i]
        conf = confidences[i]
        label = f"Class {cls} {conf:.2f}"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, label, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    return frame