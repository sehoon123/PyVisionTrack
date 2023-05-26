import cv2
import numpy as np
from Box import Box
from GestureRecognizer import GestureRecognizer
from FaceLandmarkDetector import FaceLandmarkDetector

def run_whiteboard():
    # Initialize the gesture recognizer
    recognizer = GestureRecognizer(detection_confidence=0.8)
    face_detector = FaceLandmarkDetector()

    # Initialize the video capture object
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    # create a canvas
    canvas = np.zeros((720, 1280, 3), np.uint8)

    # Previous x and y points
    px, py = 0, 0

    # Default drawing color
    color = (255, 0, 0)
    brush_size = 5
    eraser_size = 40

    # Color data
    color_data = [
        (0, 0, 100, 100, (0, 0, 255)),  # Red
        (0, 100, 100, 100, (255, 0, 0)),  # Blue
        (0, 200, 100, 100, (0, 255, 0)),  # Green
        (0, 300, 100, 100, (0, 255, 255)),  # Yellow
        (0, 400, 100, 100, (0, 0, 0), "Eraser")  # Erase (black)
    ]

    # Create color boxes
    colors = [Box(*data) for data in color_data]

    # Create clear box
    clear = Box(0, 620, 100, 100, (100, 100, 100), "Clear")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)
        frame = recognizer.detect_hands(frame)

        positions = recognizer.get_hand_position(frame)
        up_fingers = recognizer.count_up_fingers(frame)

        # initialize x, y position
        x, y = 0, 0

        if up_fingers:
            x, y = positions[8][0], positions[8][1]
            if up_fingers[1] and up_fingers[2]:
                px, py = 0, 0

                # Select color
                for cb in colors:
                    if cb.is_over(x, y):
                        color = cb.color

                # clear the canvas
                if clear.is_over(x, y):
                    canvas = np.zeros((720, 1280, 3), np.uint8)

            # Draw with index finger
            elif up_fingers[1] and not up_fingers[2]:
                if px != 0 and py != 0:
                    cv2.line(canvas, (px, py), (x, y), color, brush_size)
                px, py = x, y

            else:
                px, py = 0, 0

        # clear the canvas when middle finger is up
        if up_fingers and up_fingers[2] and not up_fingers[1] and positions:
            canvas = np.zeros((720, 1280, 3), np.uint8)

        # merge the canvas and the frame
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)

        # Draw the color boxes
        for c in colors:
            c.draw_rect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 0, 0), 2)

        clear.draw_rect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (0, 0, 0), 2)

        # Draw the pen size bar
        pen_size_bar_x = 1180
        pen_size_bar_y = 0
        pen_size_bar_width = 100
        pen_size_bar_height = 600
        pen_size_bar_color = color  # Use the selected pen color

        # Calculate the current pen size based on finger position
        if x >= pen_size_bar_x and x <= pen_size_bar_x + pen_size_bar_width:
            progress = (y - pen_size_bar_y) / pen_size_bar_height
            brush_size = int(progress * 20) + 5  # Adjust the pen size range as needed

        # Draw pen size progress bar
        cv2.rectangle(frame, (pen_size_bar_x, pen_size_bar_y), (pen_size_bar_x + pen_size_bar_width, pen_size_bar_y + pen_size_bar_height), (0,0,0), -1)
        cv2.rectangle(frame, (pen_size_bar_x, pen_size_bar_y), (pen_size_bar_x + pen_size_bar_width, int(pen_size_bar_y + pen_size_bar_height * brush_size / 25)), pen_size_bar_color, -1)  # Highlight the selected pen size based on brush_size

        # Display current pen size value
        pen_size_text = f"Pen Size: {brush_size}"
        cv2.putText(frame, pen_size_text, (pen_size_bar_x, pen_size_bar_y + pen_size_bar_height + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Detect landmarks on the face
        landmarks = face_detector.detect_faces(frame, draw=True)
        if landmarks:
            # Check if mouth is open
            eye_offset = 50
            landmark = landmarks[1]

            if face_detector.is_mouth_open(landmarks):
                cv2.putText(frame, "Mouth Open", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (int(landmark.x * 1280), int(landmark.y * 720) - 50), 200, (0, 255, 255), -1)
                cv2.rectangle(frame, (int(landmark.x * 1280) + 50, int(landmark.y * 720) - 10), (int(landmark.x * 1280) - 50, int(landmark.y * 720) + 80), (0, 0, 255), -1)
            else:
                cv2.putText(frame, "Mouth Closed", (550, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.circle(frame, (int(landmark.x * 1280), int(landmark.y * 720) - 50), 200, (0, 255, 255), -1)
                cv2.rectangle(frame, (int(landmark.x * 1280) + 50, int(landmark.y * 720) - 10), (int(landmark.x * 1280) - 50, int(landmark.y * 720) + 20), (0, 0, 255), -1)


            if face_detector.is_left_eye_open(landmarks):
                cv2.putText(frame, "Left Eye Open", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (int(landmark.x * 1280) - eye_offset, int(landmark.y * 720) - 50 - eye_offset), 20, (0, 0, 0), -1)
            else:
                cv2.putText(frame, "Left Eye Closed", (550, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(landmark.x * 1280) - eye_offset + 10, int(landmark.y * 720) - 50 - eye_offset + 10), (int(landmark.x * 1280) - eye_offset - 80, int(landmark.y * 720) - 50 - eye_offset - 10), (0, 0, 0), -1)

            if face_detector.is_right_eye_open(landmarks):
                cv2.putText(frame, "Right Eye Open", (550, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.circle(frame, (int(landmark.x * 1280) + eye_offset, int(landmark.y * 720) - 50 - eye_offset), 20, (0, 0, 0), -1)
            else:
                cv2.putText(frame, "Right Eye Closed", (550, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.rectangle(frame, (int(landmark.x * 1280) + eye_offset + 10, int(landmark.y * 720) - 50 - eye_offset + 10), (int(landmark.x * 1280) + eye_offset - 80, int(landmark.y * 720) - 50 - eye_offset - 10), (0, 0, 0), -1)

        cv2.imshow('video', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_whiteboard()
