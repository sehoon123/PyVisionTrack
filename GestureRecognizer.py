import mediapipe as mp
import cv2

class GestureRecognizer:
    def __init__(self, enable_tracking=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.enable_tracking = enable_tracking
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(self.enable_tracking, self.max_hands, int(self.detection_confidence),
                                         self.tracking_confidence)
        self.mp_draw = mp.solutions.drawing_utils
        self.results = None
        self.landmarks = {'Right': None, 'Left': None}

    def detect_hands(self, image, draw=True):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(image_rgb)

        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(image, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
        return image

    def get_hand_position(self, image, draw=True):
        landmark_list = {'Right': [], 'Left': []}
        if self.results.multi_hand_landmarks:
            for hand_num, hand_landmarks in enumerate(self.results.multi_hand_landmarks):
                hand_info = self.results.multi_handedness[hand_num]
                hand_type = 'Right' if hand_info.classification[0].label == 'Right' else 'Left'
                for landmark in hand_landmarks.landmark:
                    height, width, _ = image.shape
                    x, y = int(landmark.x * width), int(landmark.y * height)
                    landmark_list[hand_type].append((x, y))
                    if draw:
                        cv2.circle(image, (x, y), 5, (255, 0, 255), cv2.FILLED)
        return landmark_list

    def count_up_fingers(self, image):
        # Get the hand positions from the image
        positions = self.get_hand_position(image, draw=False)

        up_fingers = {'Right': [], 'Left': []}

        for hand in ['Right', 'Left']:
            # Check if positions are available
            if positions[hand]:
                # Check if each finger is up and store the results
                is_index_finger_up = positions[hand][4][1] < positions[hand][3][1] and (positions[hand][5][0] - positions[hand][4][0] > 10)
                is_middle_finger_up = positions[hand][8][1] < positions[hand][7][1] and positions[hand][7][1] < positions[hand][6][1]
                is_ring_finger_up = positions[hand][12][1] < positions[hand][11][1] and positions[hand][11][1] < positions[hand][10][1]
                is_pinky_finger_up = positions[hand][16][1] < positions[hand][15][1] and positions[hand][15][1] < positions[hand][14][1]
                is_thumb_finger_up = positions[hand][20][1] < positions[hand][19][1] and positions[hand][19][1] < positions[hand][18][1]

                # Store the results in a list
                up_fingers[hand] = [is_index_finger_up, is_middle_finger_up, is_ring_finger_up, is_pinky_finger_up,
                                    is_thumb_finger_up]

        return up_fingers
