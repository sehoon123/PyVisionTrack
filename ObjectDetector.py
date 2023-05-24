import cv2
import mediapipe as mp


class ObjectDetector:
    def __init__(self, detection_confidence=0.5):
        self.cap = None
        self.objectron = mp.solutions.objectron.Objectron(
            min_detection_confidence=detection_confidence,
            max_num_objects=5
        )

    def detect_objects(self, image, draw=True):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.objectron.process(rgb_image)

        if results.detected_objects:
            for detected_object in results.detected_objects:
                if draw:
                    mp.solutions.objectron.draw_landmarks(image, detected_object)
        return image


# Example usage
if __name__ == "__main__":
    detector = ObjectDetector()

    cap = cv2.VideoCapture(1)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        output_frame = detector.detect_objects(frame)

        cv2.imshow("Object Detection", output_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
