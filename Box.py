import cv2
import numpy as np

class Box:
    def __init__(self, x, y, w, h, color, text=''):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.text = text

    def draw_rect(self, img, text_color=(255, 255, 255), font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.8,
                  thickness=2):
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, 0, white_rect, 1, 1.0)

        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

        text_size = cv2.getTextSize(self.text, font_face, font_scale, thickness)
        text_pos = (int(self.x + self.w / 2 - text_size[0][0] / 2), int(self.y + self.h / 2 + text_size[0][1] / 2))
        cv2.putText(img, self.text, text_pos, font_face, font_scale, text_color, thickness)

    def is_over(self, x, y):
        if (self.x + self.w > x > self.x) and (self.y + self.h > y > self.y):
            return True
        return False
