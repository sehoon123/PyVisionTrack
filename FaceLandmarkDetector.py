import cv2
import mediapipe as mp


class FaceLandmarkDetector:
    def __init__(self, detection_confidence=0.5, tracking_confidence=0.5, num_faces=2, threshold=0.07):
        self.cap = None
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            min_detection_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
            max_num_faces=num_faces
        )
        self.threshold = threshold

    def detect_faces(self, image, draw=True):
        landmarks = []
        if image.ndim == 3 and image.shape[-1] == 3:
            rgb_image = image
        else:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale image to RGB

        results = self.face_mesh.process(rgb_image)

        # Check if landmarks are detected
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for i in range(0, 468):
                    landmarks.append(face_landmarks.landmark[i])

                    if draw:
                        image_height, image_width, _ = rgb_image.shape
                        x = int(face_landmarks.landmark[i].x * image_width)
                        y = int(face_landmarks.landmark[i].y * image_height)
                        cv2.circle(rgb_image, (x, y), 1, (0, 255, 0), 1)

        # Return the landmarks and the image with drawn landmarks
        return landmarks

    def is_mouth_open(self, landmarks):
        if len(landmarks) >= 68:
            mouth_height = landmarks[17].y - landmarks[13].y
            if mouth_height > self.threshold:
                return True
            else:
                return False

    def is_both_eye_open(self, landmarks):
        if len(landmarks) >= 68:
            left_eye_height = landmarks[145].y - landmarks[159].y
            print("left eye height: ", left_eye_height)
            right_eye_height = landmarks[374].y - landmarks[386].y
            print("right eye height: ", right_eye_height)
            if left_eye_height > 0.03 and right_eye_height > 0.03:
                return True
            else:
                return False
