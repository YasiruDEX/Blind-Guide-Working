import cv2 as cv

# URL for the ESP32 stream
url = "http://192.168.137.59:81/stream"

# Create a VideoCapture object
capture = cv.VideoCapture(url)

# Check if the stream is opened correctly
if not capture.isOpened():
    print("Error: Unable to open video stream")
else:
    while True:
        # Capture frame-by-frame
        ret, frame = capture.read()

        # If a frame is returned (ret is True), display it
        if ret:
            cv.imshow("ESP32 Camera Stream", frame)

            # Press 'q' to quit the streaming window
            if cv.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("Failed to retrieve frame")
            break

# Release the video capture object and close all OpenCV windows
capture.release()
cv.destroyAllWindows()
