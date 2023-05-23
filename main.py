import cv2
import numpy as np
import time
from Box import Box
from WhiteBoard import WhiteBoard
from GestureRecognizer import GestureRecognizer


def run_whiteboard():
    # Initialize the hand detector
    detector = GestureRecognizer(detection_confidence=0.8)

    # Initialize the camera
    cap = cv2.VideoCapture(1)
    cap.set(3, 1280)
    cap.set(4, 720)

    # Creating canvas to draw on
    canvas = np.zeros((720, 1280, 3), np.uint8)

    # Define a previous point to be used with drawing a line
    px, py = 0, 0

    # Initial brush color
    color = (255, 0, 0)
    brush_size = 5
    eraser_size = 20

    ########### creating colors ########

    # Define color data
    color_data = [
        (0, 0, 100, 100, (0, 0, 255)),  # Red
        (0, 100, 100, 100, (255, 0, 0)),  # Blue
        (0, 200, 100, 100, (0, 255, 0)),  # Green
        (0, 300, 100, 100, (0, 255, 255)),  # Yellow
        (0, 400, 100, 100, (0, 0, 0), "Eraser")  # Erase (black)
    ]

    # Create color boxes
    colors = [Box(*data) for data in color_data]

    # Clear
    clear = Box(0, 500, 100, 100, (100, 100, 100), "Clear")

    ########## pen sizes #######
    pens = []
    for i, pen_size in enumerate(range(5, 25, 5)):
        pens.append(Box(1180, 50 + 100 * i, 100, 100, (50, 50, 50), str(pen_size)))

    pen_btn = Box(1180, 0, 100, 50, color, 'Pen')


    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        # Call detect_hands method first
        frame = detector.detect_hands(frame)

        positions = detector.get_hand_position(frame)
        up_fingers = detector.count_up_fingers(frame)

        if up_fingers:
            x, y = positions[8][0], positions[8][1]
            if up_fingers[1] and up_fingers[2]:
                px, py = 0, 0

                ##### Pen sizes ######
                for pen in pens:
                    if pen.is_over(x, y):
                        brush_size = int(pen.text)
                        pen.alpha = 0
                    else:
                        pen.alpha = 0.5

                ####### Choose a color for drawing #######
                for cb in colors:
                    if cb.is_over(x, y):
                        color = cb.color
                        cb.alpha = 0
                    else:
                        cb.alpha = 0.5

                # Clear
                if clear.is_over(x, y):
                    clear.alpha = 0
                    canvas = np.zeros((720, 1280, 3), np.uint8)
                else:
                    clear.alpha = 0.5

                # Pen size button
                if pen_btn.is_over(x, y):
                    pen_btn.alpha = 0
                else:
                    pen_btn.alpha = 0.5

            elif up_fingers[1] and not up_fingers[2]:
                if px != 0 and py != 0:
                    cv2.line(canvas, (px, py), (x, y), color, brush_size)
                px, py = x, y

            else:
                px, py = 0, 0

        # Check if all fingers are open
        all_fingers_open = all(up_fingers)
        if all_fingers_open and positions:
            canvas = np.zeros((720, 1280, 3), np.uint8)

        # Put the white board on the frame
        canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
        _, img_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
        img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
        frame = cv2.bitwise_and(frame, img_inv)
        frame = cv2.bitwise_or(frame, canvas)

        ########## Pen colors' boxes #########
        for c in colors:
            c.draw_rect(frame)
            cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (0, 0, 0), 2)

        clear.draw_rect(frame)
        cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (0, 0, 0), 2)

        ########## Brush size boxes ######
        pen_btn.color = color
        pen_btn.draw_rect(frame)
        cv2.rectangle(frame, (pen_btn.x, pen_btn.y), (pen_btn.x + pen_btn.w, pen_btn.y + pen_btn.h), (0, 0, 0),
                      2)

        for pen in pens:
            pen.draw_rect(frame)
            cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (0, 0, 0), 2)

        cv2.imshow('video', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_whiteboard()
