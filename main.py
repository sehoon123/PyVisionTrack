import cv2
import numpy as np
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
    # Colors button
    colors_btn = Box(200, 0, 100, 100, (120, 255, 0), 'Colors')

    # Define color data
    color_data = [
        (400, 0, 100, 100, (0, 0, 255)),  # Red
        (500, 0, 100, 100, (255, 0, 0)),  # Blue
        (600, 0, 100, 100, (0, 255, 0)),  # Green
        (700, 0, 100, 100, (0, 255, 255)),  # Yellow
        (800, 0, 100, 100, (0, 0, 0), "Eraser")  # Erase (black)
    ]

    # Create color boxes
    colors = [Box(*data) for data in color_data]

    # Clear
    clear = Box(900, 0, 100, 100, (100, 100, 100), "Clear")

    ########## pen sizes #######
    pens = []
    for i, pen_size in enumerate(range(5, 25, 5)):
        pens.append(Box(1100, 50 + 100 * i, 100, 100, (50, 50, 50), str(pen_size)))

    pen_btn = Box(1100, 0, 100, 50, color, 'Pen')

    # White board button
    board_btn = Box(50, 0, 100, 100, (255, 255, 0), 'Board')

    # Define a white board to draw on
    white_board = WhiteBoard(50, 120, 1020, 580, (255, 255, 255), alpha=0.6)

    cooling_counter = 20
    hide_board = True
    hide_colors = True
    hide_pen_sizes = True

    while True:
        if cooling_counter:
            cooling_counter -= 1

        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1280, 720))
        frame = cv2.flip(frame, 1)

        # Call detect_hands method first
        frame = detector.detect_hands(frame)

        positions = detector.get_hand_position(frame, draw=False)
        up_fingers = detector.count_up_fingers(frame)

        if up_fingers:
            x, y = positions[8][0], positions[8][1]
            if up_fingers[1] and not white_board.is_over(x, y):
                px, py = 0, 0

                ##### Pen sizes ######
                if not hide_pen_sizes:
                    for pen in pens:
                        if pen.is_over(x, y):
                            brush_size = int(pen.text)
                            pen.alpha = 0
                        else:
                            pen.alpha = 0.5

                ####### Choose a color for drawing #######
                if not hide_colors:
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

                # Color button
                if colors_btn.is_over(x, y) and not cooling_counter:
                    cooling_counter = 10
                    colors_btn.alpha = 0
                    hide_colors = False if hide_colors else True
                    colors_btn.text = 'Colors' if hide_colors else 'Hide'
                else:
                    colors_btn.alpha = 0.5

                # Pen size button
                if pen_btn.is_over(x, y) and not cooling_counter:
                    cooling_counter = 10
                    pen_btn.alpha = 0
                    hide_pen_sizes = False if hide_pen_sizes else True
                    pen_btn.text = 'Pen' if hide_pen_sizes else 'Hide'
                else:
                    pen_btn.alpha = 0.5

                # White board button
                if board_btn.is_over(x, y) and not cooling_counter:
                    cooling_counter = 10
                    board_btn.alpha = 0
                    hide_board = False if hide_board else True
                    board_btn.text = 'Board' if hide_board else 'Hide'

                else:
                    board_btn.alpha = 0.5

            elif up_fingers[1] and not up_fingers[2]:
                if white_board.is_over(x, y) and not hide_board:
                    if px != 0 and py != 0:
                        cv2.line(canvas, (px, py), (x, y), color, brush_size)
                    px, py = x, y

            else:
                px, py = 0, 0

        # Check if all fingers are open
        all_fingers_open = all(up_fingers)
        if all_fingers_open and positions:
            canvas = np.zeros((720, 1280, 3), np.uint8)

        # Put colors button
        colors_btn.draw_rect(frame)
        cv2.rectangle(frame, (colors_btn.x, colors_btn.y), (colors_btn.x + colors_btn.w, colors_btn.y + colors_btn.h),
                      (255, 255, 255), 2)

        # Put white board button
        board_btn.draw_rect(frame)
        cv2.rectangle(frame, (board_btn.x, board_btn.y), (board_btn.x + board_btn.w, board_btn.y + board_btn.h),
                      (255, 255, 255), 2)

        # Put the white board on the frame
        if not hide_board:
            white_board.draw_board(frame)
            canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
            _, img_inv = cv2.threshold(canvas_gray, 20, 255, cv2.THRESH_BINARY_INV)
            img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
            frame = cv2.bitwise_and(frame, img_inv)
            frame = cv2.bitwise_or(frame, canvas)

        ########## Pen colors' boxes #########
        if not hide_colors:
            for c in colors:
                c.draw_rect(frame)
                cv2.rectangle(frame, (c.x, c.y), (c.x + c.w, c.y + c.h), (255, 255, 255), 2)

            clear.draw_rect(frame)
            cv2.rectangle(frame, (clear.x, clear.y), (clear.x + clear.w, clear.y + clear.h), (255, 255, 255), 2)

        ########## Brush size boxes ######
        pen_btn.color = color
        pen_btn.draw_rect(frame)
        cv2.rectangle(frame, (pen_btn.x, pen_btn.y), (pen_btn.x + pen_btn.w, pen_btn.y + pen_btn.h), (255, 255, 255),
                      2)
        if not hide_pen_sizes:
            for pen in pens:
                pen.draw_rect(frame)
                cv2.rectangle(frame, (pen.x, pen.y), (pen.x + pen.w, pen.y + pen.h), (255, 255, 255), 2)

        cv2.imshow('video', frame)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_whiteboard()
