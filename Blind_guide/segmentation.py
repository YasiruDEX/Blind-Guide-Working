import cv2
import pyttsx3
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors
import threading
import queue

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load YOLO segmentation model
model = YOLO("yolov8n-seg.pt")  # Ensure the correct path to your YOLOv8 model file

# Get class names
class_names = model.names

# Objects of interest
objects_of_interest = {
    "train": "Warning, a train is detected.",
    "truck": "Caution, there is a truck nearby.",
    "fire hydrant": "Fire hydrant ahead.",
    "stop sign": "Stop sign detected.",
    "dog": "Be careful, there is a dog."
}

# Open video capture (0 for default webcam)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

frame_counter = 0
skip_rate = 60  # Process every 60th frame

# Queue for TTS messages
tts_queue = queue.Queue()

def process_tts():
    while True:
        text = tts_queue.get()
        if text is None:
            break
        engine.say(text)
        engine.runAndWait()
        tts_queue.task_done()

# Start the TTS thread
tts_thread = threading.Thread(target=process_tts, daemon=True)
tts_thread.start()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    if frame_counter % skip_rate != 0:
        frame_counter += 1
        continue

    frame_counter += 1

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Perform object detection
    results = model.track(frame, persist=True)

    # Check if there are any detected objects with IDs and masks
    if results[0].boxes.id is not None and results[0].masks is not None:
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.int().cpu().tolist()

        for mask, track_id, box, cls in zip(masks, track_ids, boxes, classes):
            # Calculate the area of the bounding box (optional: can be used for distance estimation)
            box_width = box[2] - box[0]
            box_height = box[3] - box[1]
            box_area = box_width * box_height

            # Get the class name
            class_name = class_names[cls]

            # If the detected object is one of interest, add a specific message to the TTS queue
            if class_name in objects_of_interest:
                tts_queue.put(objects_of_interest[class_name])

            # Annotate the frame with the detection (optional for visual confirmation)
            annotator = Annotator(frame, line_width=2)
            annotator.seg_bbox(mask=mask, mask_color=colors(track_id, True), label=f"{track_id} {class_name}")

            frame = annotator.result()

    # Display the frame (optional for visual confirmation)
    cv2.imshow("Obstacle Detection", frame)

    # Exit if 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Stop the TTS thread
tts_queue.put(None)
tts_thread.join()

# Release resources
cap.release()
cv2.destroyAllWindows()