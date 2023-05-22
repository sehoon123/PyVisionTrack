import cv2
import numpy as np
class WhiteBoard:
    def __init__(self, x, y, w, h, color, alpha=0.5):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.color = color
        self.alpha = alpha

    def draw_board(self, img):
        alpha = self.alpha
        bg_rec = img[self.y: self.y + self.h, self.x: self.x + self.w]
        white_rect = np.ones(bg_rec.shape, dtype=np.uint8)
        white_rect[:] = self.color
        res = cv2.addWeighted(bg_rec, alpha, white_rect, 1 - alpha, 1.0)
        img[self.y: self.y + self.h, self.x: self.x + self.w] = res

    def is_over(self, x, y):
        if (self.x <= x <= self.x + self.w) and (self.y <= y <= self.y + self.h):
            return True
        return False
