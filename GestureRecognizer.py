import mediapipe as mp
import cv2

class GestureRecognizer:
    def __init__(self, enable_tracking=False, max_hands=1, detection_confidence=0.5, tracking_confidence=0.8):
        self.enable_tracking = enable_tracking
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.enable_tracking, self.max_hands, int(self.detection_confidence),
                                         self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None

    def detect_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def get_hand_position(self, image, hand_index=0, draw=True):
        landmark_list = []
        if self.results.multi_hand_landmarks:
            my_hand = self.results.multi_hand_landmarks[hand_index]
            for landmark in my_hand.landmark:
                height, width, _ = image.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                landmark_list.append((x, y))

                if draw:
                    cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)
        return landmark_list

    def count_up_fingers(self, image):
        # Get the hand positions from the image
        positions = self.get_hand_position(image, draw=False)

        # Check if positions are available
        if positions:
            # Check if each finger is up and store the results
            is_index_finger_up = positions[4][1] < positions[3][1] and (positions[5][0] - positions[4][0] > 10)
            is_middle_finger_up = positions[8][1] < positions[7][1] and positions[7][1] < positions[6][1]
            is_ring_finger_up = positions[12][1] < positions[11][1] and positions[11][1] < positions[10][1]
            is_pinky_finger_up = positions[16][1] < positions[15][1] and positions[15][1] < positions[14][1]
            is_thumb_finger_up = positions[20][1] < positions[19][1] and positions[19][1] < positions[18][1]

            # Store the results in a list
            up_fingers = [is_index_finger_up, is_middle_finger_up, is_ring_finger_up, is_pinky_finger_up,
                          is_thumb_finger_up]
        else:
            up_fingers = []

        return up_fingers
