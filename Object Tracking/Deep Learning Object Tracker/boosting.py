import cv2


def main():
    #  Initialize Boosting Tracker
    tracker = cv2.legacy.TrackerBoosting_create()

    # Open webcam or video files
    video = cv2.VideoCapture(0) #replace 0 with path to vido file 

    # Read the first frame 
    ret, frame = video.read()
    if not ret:
        print("failed to read from video video source")
        video.release()
        return

    # select ROI (object to track)
    bbox = cv2. selectROI("Select object to Track", frame , fromCenter = False, showCrosshair=True)
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
            x, y, w, h = map(int, bbox)
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)
        else:
            # Tracking failure
             cv2.putText(frame, "Tracking failure detected", (50, 80),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255,0), 2)

        # Display Result
        cv2.imshow("Boosting Tracker", frame)

        # Exit or ESC key
        key=cv2.waitKey(30) & 0xFF
        if key == 27:
            break

    # Release resources
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

