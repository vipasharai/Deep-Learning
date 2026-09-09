import cv2
import numpy as np

from ultralytics import YOLO

# load the yolo11 model
model = YOLO("yolo11n.pt")

#open the video file
#video_path = "path/to/video.mp4"
cap = cv2.VideoCapture(0)

# Loop through the Video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # run YOLO11 tracking on the frame, persisting tracks between frames
        #results = model.track(frame, persist=True)
        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # display the annotated frame
        cv2.imshow("YOLO11 Tracking", annotated_frame)

        # break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
# release the video capture object and close the display window
cap.release()
cv2.destroyAllWindows()