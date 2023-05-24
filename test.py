import cv2
from FaceLandmarkDetector import FaceLandmarkDetector


def main():
    # Initialize the FaceLandmarkDetector
    detector = FaceLandmarkDetector()

    # Open video capture
    cap = cv2.VideoCapture(1)

    while True:
        # Read frame from video capture
        ret, frame = cap.read()

        if not ret:
            break

        # Detect landmarks and check if mouth is open
        landmarks = detector.detect_faces(frame, draw=True)
        if landmarks:
            # Check if mouth is open
            if detector.is_mouth_open(landmarks):
                cv2.putText(frame, "Mouth Open", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "Mouth Closed", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if detector.is_both_eye_open(landmarks):
                cv2.putText(frame, "Both Eye Open", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "One Eye Open", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


        # Display the frame
        cv2.imshow("Face Landmark Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
