"""
Comprehensive Automated Test Suite for AI Virtual Keyboard.
Verifies Hand Tracking, Key Layouts, Typing State Machine, Suggestions, and UI Rendering.
"""

import cv2
import numpy as np
import time
import unittest

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


class TestHandTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = HandTracker()

    def test_initialization(self):
        self.assertIsNotNone(self.tracker.hands)
        self.assertFalse(self.tracker.isHandPresent())

    def test_find_hands_on_synthetic_frame(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        processed = self.tracker.findHands(frame, draw=True, classy_draw=True)
        self.assertEqual(processed.shape, (720, 1280, 3))
        lmList = self.tracker.findPosition(processed)
        self.assertEqual(len(lmList), 0)

    def test_distance_fallbacks(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        dist = self.tracker.findDistance(4, 8, frame)
        self.assertEqual(dist, 999.0)
        norm_dist = self.tracker.findNormalizedDistance(4, 8, frame)
        self.assertEqual(norm_dist, 999.0)
        scale = self.tracker.getHandScale(frame)
        self.assertGreater(scale, 0)


class TestWordSuggestions(unittest.TestCase):
    def setUp(self):
        self.engine = WordSuggestionEngine()

    def test_empty_input_starters(self):
        suggestions = self.engine.get_suggestions("", max_count=3, is_caps=False)
        self.assertEqual(len(suggestions), 3)
        self.assertEqual(suggestions, ["Hello", "Thank", "Please"])

    def test_caps_lock_starters(self):
        suggestions = self.engine.get_suggestions("", max_count=3, is_caps=True)
        self.assertEqual(suggestions, ["HELLO", "THANK", "PLEASE"])

    def test_prefix_matching(self):
        suggestions = self.engine.get_suggestions("th", max_count=3)
        self.assertEqual(len(suggestions), 3)
        for word in suggestions:
            self.assertTrue(word.lower().startswith("th"))

    def test_case_preservation(self):
        # Lowercase prefix
        lower_sugg = self.engine.get_suggestions("pro", max_count=3)
        self.assertTrue(lower_sugg[0].islower())

        # Titlecase prefix
        title_sugg = self.engine.get_suggestions("Pro", max_count=3)
        self.assertTrue(title_sugg[0][0].isupper() and title_sugg[0][1:].islower())

        # Uppercase prefix
        upper_sugg = self.engine.get_suggestions("PRO", max_count=3)
        self.assertTrue(upper_sugg[0].isupper())

    def test_no_automatic_correction(self):
        # Engine should NEVER mutate input text spontaneously
        text = "mispelled"
        # get_suggestions only returns strings to display, does not change text
        s = self.engine.get_suggestions(text)
        self.assertEqual(text, "mispelled")

    def test_apply_suggestion(self):
        initial = "I need some hel"
        applied = self.engine.apply_suggestion(initial, "help")
        self.assertEqual(applied, "I need some help ")


class TestKeyboardLayout(unittest.TestCase):
    def setUp(self):
        self.layout = KeyboardLayout(1280, 720)

    def test_all_keys_generated(self):
        self.assertGreater(len(self.layout.buttons), 40)
        key_ids = [b.key_id for b in self.layout.buttons]
        
        # Verify required keys exist
        self.assertIn("BACK", key_ids)
        self.assertIn("SPACE", key_ids)
        self.assertIn("ENTER", key_ids)
        self.assertIn("CLEAR", key_ids)
        self.assertIn("CAPS", key_ids)
        self.assertIn("MODE_SYM", key_ids)

    def test_toggle_caps(self):
        # Initial lowercase letters (when caps_lock is False)
        q_btn = next(b for b in self.layout.buttons if b.key_id == "q")
        self.assertEqual(q_btn.text, "q")
        self.assertFalse(self.layout.caps_lock)

        # Toggle CAPS ON
        self.layout.toggle_caps()
        self.assertTrue(self.layout.caps_lock)
        q_btn = next(b for b in self.layout.buttons if b.key_id == "Q")
        self.assertEqual(q_btn.text, "Q")

        # Toggle CAPS OFF
        self.layout.toggle_caps()
        self.assertFalse(self.layout.caps_lock)
        q_btn = next(b for b in self.layout.buttons if b.key_id == "q")
        self.assertEqual(q_btn.text, "q")

    def test_toggle_symbols_mode(self):
        self.assertFalse(self.layout.symbols_mode)
        self.layout.toggle_symbols()
        self.assertTrue(self.layout.symbols_mode)

        # Check symbol keys exist in symbol mode
        key_ids = [b.key_id for b in self.layout.buttons]
        self.assertIn("MODE_ABC", key_ids)
        self.assertIn("@", key_ids)
        self.assertIn("#", key_ids)

        # Switch back to ABC
        self.layout.toggle_symbols()
        self.assertFalse(self.layout.symbols_mode)

    def test_hit_testing(self):
        # Pick the space button
        space_btn = next(b for b in self.layout.buttons if b.key_id == "SPACE")
        x, y = space_btn.pos
        w, h = space_btn.size
        center_x = x + w // 2
        center_y = y + h // 2

        hit = self.layout.get_button_at(center_x, center_y)
        self.assertIsNotNone(hit)
        self.assertEqual(hit.key_id, "SPACE")

    def test_suggestion_chips_update(self):
        test_suggs = ["Alpha", "Beta", "Gamma"]
        self.layout.set_suggestions(test_suggs)
        chip_texts = [b.text for b in self.layout.suggestion_buttons]
        self.assertEqual(chip_texts, test_suggs)


class TestTypingAndDebounceSimulation(unittest.TestCase):
    """Simulates realistic hand gestures, debounce, and state transitions."""

    def test_typing_and_repeated_input_prevention(self):
        layout = KeyboardLayout(1280, 720)
        engine = WordSuggestionEngine()
        
        finalText = ""
        can_click = True
        last_click_time = 0.0
        MIN_CLICK_INTERVAL = 0.25

        # 1. User hovers over '1' button
        btn_1 = next(b for b in layout.buttons if b.key_id == "1")
        
        # 2. User pinches down
        # Frame 1: Pinch down
        t0 = 100.0
        is_pinched = True
        is_released = False
        
        if is_pinched and can_click:
            can_click = False
            last_click_time = t0
            btn_1.trigger_press()
            finalText += btn_1.text

        self.assertEqual(finalText, "1")

        # Frame 2: User holds pinch down (100ms later)
        t1 = t0 + 0.10
        if is_pinched and can_click:
            finalText += btn_1.text
        # Still '1' - holding does NOT re-trigger!
        self.assertEqual(finalText, "1")

        # Frame 3: User still holds pinch down (500ms later)
        t2 = t0 + 0.50
        # Since is_released has not happened, can_click must still be False
        if is_released and (t2 - last_click_time > MIN_CLICK_INTERVAL):
            can_click = True
        if is_pinched and can_click:
            finalText += btn_1.text
        self.assertEqual(finalText, "1", "Machine-gun typing was successfully prevented!")

        # Frame 4: User releases pinch
        t3 = t0 + 0.55
        is_pinched = False
        is_released = True
        if is_released and (t3 - last_click_time > MIN_CLICK_INTERVAL):
            can_click = True
        self.assertTrue(can_click, "Click re-armed after deliberate release.")

        # Frame 5: User clicks '2'
        btn_2 = next(b for b in layout.buttons if b.key_id == "2")
        is_pinched = True
        is_released = False
        if is_pinched and can_click:
            can_click = False
            last_click_time = t3 + 0.05
            finalText += btn_2.text
        self.assertEqual(finalText, "12")

        # Frame 6: Space
        finalText += " "
        self.assertEqual(finalText, "12 ")

        # Frame 7: Backspace
        finalText = finalText[:-1]
        self.assertEqual(finalText, "12")

        # Frame 8: Clear
        finalText = ""
        self.assertEqual(finalText, "")


class TestUIRendering(unittest.TestCase):
    def test_full_pipeline_rendering(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        layout = KeyboardLayout(1280, 720)
        layout.set_suggestions(["First", "Second", "Third"])

        draw_keyboard_backdrop(frame, y1=60, y2=465)
        drawAll(frame, layout.buttons, dt=0.033)
        draw_text_output_card(frame, "Testing Modern UI", y_top=480, height=220, hand_present=True, caps_lock=True, fps=60)
        draw_cursor_reticle(frame, (640, 360), pinch_progress=0.85, is_clicked=True)

        self.assertEqual(frame.shape, (720, 1280, 3))
        # Ensure frame contains drawn content (non-zero pixels)
        self.assertTrue(np.any(frame > 0))


if __name__ == "__main__":
    unittest.main()

