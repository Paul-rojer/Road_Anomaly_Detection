import cv2
import datetime
import time
from inference import InferenceModel
from postprocess import filter_boxes
from utils import draw_boxes

VIDEO_PATH = "../datasets/pothole_test.mp4"
OUTPUT_PATH = "../Final_output/Output.mp4"
MODEL_PATH = "../model/best.onnx"
INPUT_SIZE = 320
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_input = cap.get(cv2.CAP_PROP_FPS)
    if fps_input == 0 or fps_input is None:
        fps_input = 20.0
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps_input, (width, height))

    model = InferenceModel(MODEL_PATH, INPUT_SIZE)
    print("Starting Raspberry Pi Inference...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        start_time = time.time()
        predictions = model.predict(frame)
        boxes, confidences, class_ids = filter_boxes(predictions, width, height, INPUT_SIZE, CONF_THRESHOLD)

        # NMS
        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, IOU_THRESHOLD)
        if len(indices) > 0:
            boxes = [boxes[i[0]] for i in indices]
            confidences = [confidences[i[0]] for i in indices]
            class_ids = [class_ids[i[0]] for i in indices]

        frame = draw_boxes(frame, boxes, class_ids, confidences)

        # Timestamp and FPS
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        timestamp = datetime.datetime.now().strftime("%d-%m-%Y %H:%M:%S")
        cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(frame, f"FPS: {int(fps)}", (width-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)

        out.write(frame)
        cv2.imshow("Road Anomaly Detection - Raspberry Pi 4", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"Inference Completed. Output saved at {OUTPUT_PATH}")

if __name__ == "__main__":
    main()