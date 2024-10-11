import requests
import cv2 as cv
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import numpy as np

# Initialize the BLIP processor and model for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# URL for the ESP32 stream (replace with your stream URL)
url = "http://192.168.137.139:81/stream"

# Create a VideoCapture object for the stream
capture = cv.VideoCapture(url)

# Check if the stream is opened correctly
if not capture.isOpened():
    print("Error: Unable to open video stream")
else:
    frame_count = 0
    frame_skip = 5  # Number of frames to skip before generating a caption

    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        if ret:
            # Resize the frame to make processing faster
            small_frame = cv.resize(frame, (320, 240))

            # Display the video stream
            cv.imshow("ESP32 Camera Stream", small_frame)

            # Only process every nth frame to avoid buffering
            if frame_count % frame_skip == 0:
                # Convert the OpenCV frame (BGR) to PIL image (RGB)
                pil_image = Image.fromarray(cv.cvtColor(small_frame, cv.COLOR_BGR2RGB))

                # Unconditional image captioning
                inputs = processor(pil_image, return_tensors="pt")
                out = model.generate(**inputs)

                # Decode and print the generated caption
                caption = processor.decode(out[0], skip_special_tokens=True)
                print(f"Caption: {caption}")

            frame_count += 1

            # Press 'q' to quit the streaming window
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to retrieve frame")
            break

# Release the video capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
