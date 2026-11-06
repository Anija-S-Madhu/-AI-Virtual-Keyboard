"""
AI Virtual Keyboard - Modern Edition
Main application loop integrating Hand Tracking, Modern Keyboard UI,
Scale-Invariant Gesture Recognition, and Word Suggestions.
"""

import cv2
import time
import sys
from typing import Optional

from handTracker import HandTracker
from keyboard import (
    Button,
    KeyboardLayout,
    drawAll,
    draw_keyboard_backdrop,
    draw_text_output_card,
    draw_cursor_reticle,
    THEME
)
from word_suggestions import WordSuggestionEngine


def initialize_camera(width: int = 1280, height: int = 720) -> Optional[cv2.VideoCapture]:
    """Attempts to open the best available camera backend."""
    # Try standard default index
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        # Try DirectShow backend on Windows
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        # Try index 1
        cap = cv2.VideoCapture(1)

    if cap.isOpened():
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)
        return cap
    return None


def main():
    # 1. Initialize Camera
    cap = initialize_camera(1280, 720)
    using_mock_camera = False

    if cap is None:
        print("[WARNING] No physical camera found or camera cannot be accessed in current environment.")
        print("[INFO] Starting Virtual Keyboard in Simulation / Demo mode.")
        using_mock_camera = True

    # 2. Hand Tracker & Keyboard Engine
    tracker = HandTracker(detection_con=0.75, tracking_con=0.75)
    layout = KeyboardLayout(screen_w=1280, screen_h=720)
    suggestion_engine = WordSuggestionEngine()

    # 3. State Variables
    finalText = ""
    last_click_time = 0.0
    can_click = True
    locked_button: Optional[Button] = None
    last_frame_time = time.time()
    
    # Pinch Thresholds (Normalized to hand scale + pixel safety bounds)
    CLICK_NORM_THRESH = 0.30
    RELEASE_NORM_THRESH = 0.42
    CLICK_RAW_THRESH = 42
    RELEASE_RAW_THRESH = 55
    MIN_CLICK_INTERVAL = 0.28  # Debounce interval in seconds

    print("==================================================")
    print("   AI VIRTUAL KEYBOARD - MODERN EDITION RUNNING   ")
    print("==================================================")
    print(" Controls:")
    print(" - Hover index finger over any key")
    print(" - Pinch index finger and thumb to click")
    print(" - Tap suggestion chips to insert words")
    print(" - Press 'q' or ESC in window to exit")
    print("==================================================")

    while True:
        current_time = time.time()
        dt = max(0.001, min(0.1, current_time - last_frame_time))
        last_frame_time = current_time
        fps = int(1.0 / dt)

        # Frame Acquisition
        if not using_mock_camera:
            success, img = cap.read()
            if not success or img is None:
                # If frame reading fails, pause briefly and retry
                time.sleep(0.01)
                continue
        else:
            # Generate a clean dark synthetic studio background for headless/demo testing
            import numpy as np
            img = np.zeros((720, 1280, 3), dtype=np.uint8)
            img[:] = (20, 24, 32)  # Ambient studio dark slate

        # Mirror image for natural intuitive interaction
        img = cv2.flip(img, 1)

        # Hand Detection & Tracking
        img = tracker.findHands(img, draw=True, classy_draw=True)
        lmList = tracker.findPosition(img)
        hand_present = tracker.isHandPresent()

        # Update Suggestions based on currently typed text
        current_suggestions = suggestion_engine.get_suggestions(
            finalText, max_count=3, is_caps=layout.caps_lock
        )
        layout.set_suggestions(current_suggestions)

        # Render floating keyboard backdrop card
        draw_keyboard_backdrop(img, y1=60, y2=465)

        # Cursor and Gesture Processing
        smoothed_tip = tracker.getSmoothedIndexTip(img)
        hovered_button: Optional[Button] = None
        pinch_progress = 0.0
        is_pinched = False
        is_released = True

        if hand_present and smoothed_tip is not None and len(lmList) > 8:
            cx, cy = smoothed_tip
            raw_dist = tracker.findDistance(4, 8, img)
            hand_scale = tracker.getHandScale(img)
            norm_dist = raw_dist / max(hand_scale, 1.0)

            # Compute normalized pinch progress [0.0 = fully open, 1.0 = clicked]
            pinch_progress = max(0.0, min(1.0, (RELEASE_NORM_THRESH - norm_dist) / (RELEASE_NORM_THRESH - CLICK_NORM_THRESH)))

            # State check
            is_pinched = (norm_dist <= CLICK_NORM_THRESH) or (raw_dist <= CLICK_RAW_THRESH)
            is_released = (norm_dist >= RELEASE_NORM_THRESH) and (raw_dist >= RELEASE_RAW_THRESH)

            # Find key under cursor
            detected_button = layout.get_button_at(cx, cy)

            # Target Locking: Lock onto the hovered key when pinch begins to prevent
            # slight hand shift during pinch motion from triggering neighboring keys
            if pinch_progress > 0.4 and locked_button is not None:
                hovered_button = locked_button
            else:
                hovered_button = detected_button
                if pinch_progress > 0.25:
                    locked_button = detected_button

            # Reset lock when fingers open
            if is_released:
                locked_button = None

            # Re-arm click when pinch is released and debounce time elapsed
            if is_released and (current_time - last_click_time > MIN_CLICK_INTERVAL):
                can_click = True

            # Register Click Event
            if is_pinched and can_click and hovered_button is not None:
                can_click = False
                last_click_time = current_time
                hovered_button.trigger_press()

                # Process Key Press Action
                key_id = hovered_button.key_id

                if key_id == "BACK":
                    # Delete last character
                    finalText = finalText[:-1]

                elif key_id == "SPACE":
                    # Append space
                    finalText += " "

                elif key_id == "ENTER":
                    # Output newline
                    finalText += "\n"

                elif key_id == "CLEAR":
                    # Clear entire text buffer
                    finalText = ""

                elif key_id == "CAPS":
                    # Toggle CAPS Lock
                    layout.toggle_caps()

                elif key_id in ("MODE_SYM", "MODE_ABC"):
                    # Toggle between letters and symbols
                    layout.toggle_symbols()

                elif hovered_button.key_type == "suggestion":
                    # Explicit manual suggestion selection (NEVER auto-replaces)
                    if hovered_button.text:
                        finalText = suggestion_engine.apply_suggestion(finalText, hovered_button.text)

                else:
                    # Regular character input: preserve exact typed character
                    finalText += hovered_button.text

        # Update button hover states
        for btn in layout.buttons:
            btn.is_hovered = (btn == hovered_button)

        # Draw Keyboard Buttons
        img = drawAll(img, layout.buttons, dt=dt)

        # Draw Bottom Text Output Card
        draw_text_output_card(
            img,
            text=finalText,
            y_top=480,
            height=220,
            hand_present=hand_present,
            caps_lock=layout.caps_lock,
            fps=fps
        )

        # Draw Cursor Reticle at index tip
        if hand_present and smoothed_tip is not None:
            is_just_clicked = (current_time - last_click_time < 0.15)
            draw_cursor_reticle(img, smoothed_tip, pinch_progress=pinch_progress, is_clicked=is_just_clicked)

        # Display Result
        cv2.imshow("AI Virtual Keyboard - Modern Edition", img)

        # Key listener (ESC or 'q' to quit)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
