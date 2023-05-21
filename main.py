import cv2

# Choose the tracking algorithm
tracker_type = "CSRT"  # Change this to the desired algorithm (KCF, MOSSE, CSRT)

# Initialize the tracker
tracker = cv2.TrackerCSRT_create()  # Change the tracker based on the chosen algorithm

# Read the video file or start the camera feed
video_path = 0  # Change this to the video file path or 0 for live camera feed
video_capture = cv2.VideoCapture(video_path)

# Read the first frame
ret, frame = video_capture.read()
if not ret:
    print("Error reading the video file or starting the camera feed.")
    exit()

# Select the object to track using the graphical interface (optional)
bounding_box = cv2.selectROI("Select Object", frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Initialize the tracker with the first frame and bounding box
tracker.init(frame, bounding_box)

while True:
    # Read a new frame
    ret, frame = video_capture.read()
    if not ret:
        break

    # Update the tracker
    success, bounding_box = tracker.update(frame)

    # Draw bounding box around the tracked object
    if success:
        (x, y, w, h) = [int(v) for v in bounding_box]
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

    # Display the resulting frame
    cv2.imshow("Object Tracking", frame)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
