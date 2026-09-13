# Combination of Boosting and MIL. 

import cv2


#  Initialize median flow Tracker
tracker_type = 'medianflow'
tracker = cv2.legacy.TrackerMedianFlow_create()

# Open webcam or video files
video = cv2.VideoCapture(0) #replace 0 with path to vido file 

# Read the first frame 
ret, frame = video.read()
if not ret:
    print("failed to read video ")
    exit()

# select ROI (object to track)
bbox = cv2. selectROI("Select object ", frame , fromCenter = False, showCrosshair=True)
cv2.destroyAllWindows()

# Initialize the Traker with the selected ROI
tracker.init(frame, bbox)

while True:
    ret, frame = video.read()
    if not ret:
        break

        # update the tracker
    success, bbox = tracker.update(frame)

    if success:
    # Tracking success: draw bounding box
        x, y, w, h = [int(v) for v in bbox]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2, 1)
        cv2.putText(frame, "Tracking", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)
    else:
    # Tracking failure
        cv2.putText(frame, "Lost", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)

        # Display Result
    cv2.imshow("Tracking", frame)

        # Exit or ESC key
    key=cv2.waitKey(30) & 0xFF
    if key == 27:
        break

# Release resources
video.release()
cv2.destroyAllWindows()



