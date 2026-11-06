"""
Hand Tracker Module with MediaPipe and OpenCV.
Provides accurate landmark detection, cursor smoothing, distance normalization,
and elegant modern hand visualization.
"""

import cv2
import mediapipe as mp
import math
from typing import List, Tuple, Optional


class HandTracker:
    """Accurate hand tracker preserving existing interface with modern enhancements."""

    def __init__(self, mode: bool = False, max_hands: int = 1, detection_con: float = 0.7, tracking_con: float = 0.7):
        self.mode = mode
        self.max_hands = max_hands
        self.detection_con = detection_con
        self.tracking_con = tracking_con

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_con,
            min_tracking_confidence=self.tracking_con
        )

        self.mpDraw = mp.solutions.drawing_utils
        self.results = None
        self.lmList = []
        
        # Cursor smoothing state (Exponential Moving Average)
        self.smooth_x = None
        self.smooth_y = None
        self.smoothing_factor = 0.5  # Responsive yet jitter-free

    def findHands(self, img, draw: bool = True, classy_draw: bool = True):
        """
        Processes image to detect hands.
        If draw=True and classy_draw=True, draws a refined, elegant cyberpunk-style hand skeleton.
        If classy_draw=False, draws default MediaPipe landmarks.
        """
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        
        # Reset cached landmarks for this new frame
        self.lmList = []

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    if classy_draw:
                        self._draw_classy_hand(img, handLms)
                    else:
                        self.mpDraw.draw_landmarks(
                            img,
                            handLms,
                            self.mpHands.HAND_CONNECTIONS
                        )

        return img

    def _draw_classy_hand(self, img, handLms):
        """Draws subtle, modern, minimalist hand landmarks and connections."""
        h, w, _ = img.shape
        coords = []
        for lm in handLms.landmark:
            cx, cy = int(lm.x * w), int(lm.y * h)
            coords.append((cx, cy))

        # Hand skeleton connections (standard MediaPipe pairs)
        connections = self.mpHands.HAND_CONNECTIONS

        # Draw soft connections (cyber-cyan/slate subtle aesthetic)
        for p1_id, p2_id in connections:
            pt1 = coords[p1_id]
            pt2 = coords[p2_id]
            cv2.line(img, pt1, pt2, (180, 140, 60), 1, cv2.LINE_AA)

        # Draw subtle joint dots
        for idx, (cx, cy) in enumerate(coords):
            if idx in (4, 8):  # Thumb and Index tips (primary interaction fingers)
                cv2.circle(img, (cx, cy), 5, (255, 200, 50), -1, cv2.LINE_AA)
                cv2.circle(img, (cx, cy), 7, (255, 240, 150), 1, cv2.LINE_AA)
            else:
                cv2.circle(img, (cx, cy), 3, (140, 110, 50), -1, cv2.LINE_AA)

    def findPosition(self, img, draw: bool = False) -> List[List[int]]:
        """
        Extracts landmark positions: [[id, cx, cy], ...]
        Preserves original signature and return format.
        """
        if self.lmList and len(self.lmList) > 0:
            return self.lmList

        self.lmList = []

        if self.results and self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[0]
            h, w, _ = img.shape

            for lm_id, lm in enumerate(hand.landmark):
                cx = int(lm.x * w)
                cy = int(lm.y * h)
                self.lmList.append([lm_id, cx, cy])

                if draw and lm_id == 8:
                    cv2.circle(img, (cx, cy), 10, (255, 180, 50), cv2.FILLED)

        return self.lmList

    def isHandPresent(self) -> bool:
        """Returns True if a hand is currently tracked."""
        return bool(self.results and self.results.multi_hand_landmarks)

    def findDistance(self, p1: int, p2: int, img=None) -> float:
        """
        Calculates euclidean pixel distance between landmarks p1 and p2.
        Preserves original behavior.
        """
        if img is not None and not self.lmList:
            lmList = self.findPosition(img)
        else:
            lmList = self.lmList

        if len(lmList) == 0 or max(p1, p2) >= len(lmList):
            return 999.0

        x1, y1 = lmList[p1][1], lmList[p1][2]
        x2, y2 = lmList[p2][1], lmList[p2][2]

        return math.hypot(x2 - x1, y2 - y1)

    def getHandScale(self, img=None) -> float:
        """
        Returns reference hand scale based on distance between wrist (0)
        and middle finger MCP (9). Scale invariant to finger pinch.
        """
        if img is not None and not self.lmList:
            lmList = self.findPosition(img)
        else:
            lmList = self.lmList

        if len(lmList) < 10:
            return 100.0  # Safe fallback

        x0, y0 = lmList[0][1], lmList[0][2]
        x9, y9 = lmList[9][1], lmList[9][2]
        scale = math.hypot(x9 - x0, y9 - y0)
        return max(scale, 20.0)

    def findNormalizedDistance(self, p1: int, p2: int, img=None) -> float:
        """
        Calculates scale-invariant normalized distance between landmarks.
        Returns distance relative to the user's hand size.
        """
        raw_dist = self.findDistance(p1, p2, img)
        if raw_dist >= 999.0:
            return 999.0
        scale = self.getHandScale(img)
        return raw_dist / scale

    def getSmoothedIndexTip(self, img=None) -> Optional[Tuple[int, int]]:
        """
        Returns smoothed (cx, cy) coordinates for index finger tip (landmark 8)
        to eliminate camera noise and micro-jitter.
        """
        if img is not None and not self.lmList:
            lmList = self.findPosition(img)
        else:
            lmList = self.lmList

        if len(lmList) <= 8:
            self.smooth_x = None
            self.smooth_y = None
            return None

        raw_x = lmList[8][1]
        raw_y = lmList[8][2]

        if self.smooth_x is None or self.smooth_y is None:
            self.smooth_x = float(raw_x)
            self.smooth_y = float(raw_y)
        else:
            alpha = self.smoothing_factor
            self.smooth_x = alpha * raw_x + (1.0 - alpha) * self.smooth_x
            self.smooth_y = alpha * raw_y + (1.0 - alpha) * self.smooth_y

        return int(self.smooth_x), int(self.smooth_y)
