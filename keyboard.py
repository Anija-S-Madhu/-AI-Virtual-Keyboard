"""
Modern Virtual Keyboard UI Module.
Features sophisticated dark glassmorphism, rounded keys, smooth animation interpolation,
dynamic text alignment, and full layout control (Numbers, QWERTY, Symbols, Special keys, Suggestions).
"""

import cv2
import numpy as np
import time
from typing import List, Tuple, Optional


# Sophisticated Dark Theme Color Palette (BGR format for OpenCV)
THEME = {
    # Panel & Backdrop
    "backdrop_bg": (14, 17, 24),           # Deep obsidian glass
    "backdrop_border": (48, 56, 75),       # Subtle perimeter slate
    
    # Standard Keys
    "key_bg_normal": (28, 33, 46),         # Dark graphite slate
    "key_border_normal": (58, 68, 90),     # Refined border
    "key_text_normal": (240, 245, 252),    # Crisp soft white
    
    # Action Keys (Space, Back, Enter, Caps, etc.)
    "key_bg_action": (38, 44, 62),         # Muted indigo slate
    "key_border_action": (75, 88, 118),    # Indigo highlight
    "key_text_action": (245, 248, 255),
    
    # Hover State
    "key_bg_hover": (48, 64, 92),          # Illuminated slate
    "key_border_hover": (255, 210, 60),    # Electric Azure/Cyan border
    "key_text_hover": (255, 255, 255),     # Brilliant white
    
    # Press State
    "key_bg_press": (210, 140, 30),        # Vibrant electric blue/azure flash
    "key_border_press": (255, 255, 255),   # Luminous white perimeter
    "key_text_press": (255, 255, 255),     # Brilliant crisp white
    
    # Active/Toggle Lock State (Caps Lock, Symbol Lock)
    "key_active_dot": (255, 215, 60),      # Cyan indicator dot
    "key_active_border": (255, 215, 60),
    
    # Suggestion Chips
    "sugg_bg_normal": (30, 37, 52),        # Floating card slate
    "sugg_border_normal": (80, 96, 126),   # Subtle cyan tint
    "sugg_text_normal": (230, 240, 255),
    "sugg_bg_hover": (45, 65, 96),
    "sugg_border_hover": (255, 215, 60),
    "sugg_badge": (180, 140, 50),
    
    # Status Indicators
    "status_emerald": (90, 220, 110),      # Hand active
    "status_amber": (50, 160, 245),        # Warning / Searching
    "status_dim": (110, 120, 140)
}


def lerp_color(c1: Tuple[int, int, int], c2: Tuple[int, int, int], t: float) -> Tuple[int, int, int]:
    """Linear interpolation between two BGR color tuples."""
    t = max(0.0, min(1.0, t))
    return (
        int(c1[0] + (c2[0] - c1[0]) * t),
        int(c1[1] + (c2[1] - c1[1]) * t),
        int(c1[2] + (c2[2] - c1[2]) * t)
    )


def draw_rounded_rect(img, pt1: Tuple[int, int], pt2: Tuple[int, int], 
                      color: Tuple[int, int, int], radius: int = 10, thickness: int = -1):
    """
    Draws a smooth rounded rectangle on an image with anti-aliasing.
    Supports filled (-1) and stroked (thickness >= 1).
    """
    x1, y1 = pt1
    x2, y2 = pt2
    w = x2 - x1
    h = y2 - y1
    r = max(1, min(radius, w // 2, h // 2))

    if thickness < 0:
        # Filled rounded rectangle
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1, lineType=cv2.LINE_AA)
    else:
        # Stroked rounded rectangle
        cv2.line(img, (x1 + r, y1), (x2 - r, y1), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1 + r, y2), (x2 - r, y2), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x1, y1 + r), (x1, y2 - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.line(img, (x2, y1 + r), (x2, y2 - r), color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness, lineType=cv2.LINE_AA)
        cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness, lineType=cv2.LINE_AA)


class Button:
    """
    Modern Interactive Virtual Keyboard Button.
    Maintains exact pos, size, and text attributes for full backward compatibility,
    while adding smooth hover/press animations and sophisticated styling.
    """

    def __init__(self, pos: List[int], text: str, size: List[int] = None, 
                 key_type: str = "letter", key_id: Optional[str] = None, radius: int = 10):
        if size is None:
            size = [70, 70]
        self.pos = pos
        self.size = size
        self.text = text
        self.key_type = key_type  # 'letter', 'number', 'action', 'space', 'suggestion', 'toggle'
        self.key_id = key_id or text
        self.radius = radius

        # State & Animation Variables
        self.is_hovered = False
        self.is_pressed = False
        self.is_active = False       # e.g., Caps Lock enabled
        self.anim_hover = 0.0        # Smooth float 0.0 -> 1.0
        self.anim_press = 0.0        # Smooth float 1.0 -> 0.0 decay
        self.last_anim_time = time.time()

    def contains(self, x: int, y: int) -> bool:
        """Checks if a point (x, y) is within the button bounds."""
        bx, by = self.pos
        bw, bh = self.size
        return bx <= x <= bx + bw and by <= y <= by + bh

    def trigger_press(self):
        """Triggers a tactile press animation."""
        self.anim_press = 1.0
        self.is_pressed = True

    def update_animation(self, is_hovered: bool, dt: float = 0.033):
        """Smoothly updates hover and press animation states."""
        self.is_hovered = is_hovered
        target_hover = 1.0 if is_hovered else 0.0
        
        # Exponential smoothing for hover
        hover_speed = 12.0
        self.anim_hover += (target_hover - self.anim_hover) * min(1.0, hover_speed * dt)

        # Decay press flash
        if self.anim_press > 0:
            press_decay_speed = 8.0
            self.anim_press = max(0.0, self.anim_press - press_decay_speed * dt)
            if self.anim_press == 0:
                self.is_pressed = False

    def draw(self, img, dt: float = 0.033):
        """Renders the button with modern dark theme and smooth animations."""
        self.update_animation(self.is_hovered, dt)

        x, y = self.pos
        w, h = self.size
        x2, y2 = x + w, y + h
        radius = min(self.radius, w // 2, h // 2)

        # Base color selection
        if self.key_type == "suggestion":
            base_bg = THEME["sugg_bg_normal"]
            hover_bg = THEME["sugg_bg_hover"]
            base_border = THEME["sugg_border_normal"]
            hover_border = THEME["sugg_border_hover"]
            base_text = THEME["sugg_text_normal"]
        elif self.key_type in ("action", "toggle", "space"):
            base_bg = THEME["key_bg_action"]
            hover_bg = THEME["key_bg_hover"]
            base_border = THEME["key_border_action"]
            hover_border = THEME["key_border_hover"]
            base_text = THEME["key_text_action"]
        else:
            base_bg = THEME["key_bg_normal"]
            hover_bg = THEME["key_bg_hover"]
            base_border = THEME["key_border_normal"]
            hover_border = THEME["key_border_hover"]
            base_text = THEME["key_text_normal"]

        # Active lock override (e.g. CAPS active)
        if self.is_active:
            base_border = THEME["key_active_border"]

        # Blend colors based on hover animation
        current_bg = lerp_color(base_bg, hover_bg, self.anim_hover)
        current_border = lerp_color(base_border, hover_border, self.anim_hover)
        current_text_color = lerp_color(base_text, THEME["key_text_hover"], self.anim_hover)

        # Blend colors on press flash
        if self.anim_press > 0.01:
            current_bg = lerp_color(current_bg, THEME["key_bg_press"], self.anim_press)
            current_border = lerp_color(current_border, THEME["key_border_press"], self.anim_press)
            current_text_color = lerp_color(current_text_color, THEME["key_text_press"], self.anim_press)

        # 1. Draw Key Fill
        draw_rounded_rect(img, (x, y), (x2, y2), current_bg, radius=radius, thickness=-1)

        # 2. Draw Key Border
        border_thickness = 2 if (self.anim_hover > 0.3 or self.is_active) else 1
        draw_rounded_rect(img, (x, y), (x2, y2), current_border, radius=radius, thickness=border_thickness)

        # 3. Draw Active indicator dot for toggle keys (like CAPS lock)
        if self.is_active and self.key_type == "toggle":
            cv2.circle(img, (x + 12, y + 12), 4, THEME["key_active_dot"], -1, cv2.LINE_AA)

        # 4. Draw Typography with dynamic centering
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Determine appropriate font scale based on text length and button size
        label = self.text
        if len(label) == 1:
            font_scale = 0.72
            font_thickness = 2
        elif len(label) <= 4:
            font_scale = 0.54
            font_thickness = 2
        elif len(label) <= 8:
            font_scale = 0.50
            font_thickness = 2
        else:
            font_scale = 0.44
            font_thickness = 1

        (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Center coordinates
        tx = x + (w - text_w) // 2
        ty = y + (h + text_h) // 2 - 1

        cv2.putText(
            img,
            label,
            (tx, ty),
            font,
            font_scale,
            current_text_color,
            font_thickness,
            cv2.LINE_AA
        )


class KeyboardLayout:
    """
    Manages the complete virtual keyboard layout:
    - Number row (1-0)
    - Letter rows (QWERTY)
    - Function keys: CAPS, BACKSPACE, SPACE, ENTER, CLEAR, ?123 Mode
    - Symbol layout mode
    - Word suggestion chips bar
    """

    def __init__(self, screen_w: int = 1280, screen_h: int = 720):
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.caps_lock = False
        self.symbols_mode = False
        self.buttons: List[Button] = []
        self.suggestion_buttons: List[Button] = []
        
        self.rebuild_layout()

    def rebuild_layout(self):
        """Constructs all buttons with exact alignment, spacing, and centered layout."""
        self.buttons = []
        self.suggestion_buttons = []

        # 1. Build Suggestion Bar (Above keyboard, y = 70 to 118)
        sugg_w = 270
        sugg_h = 44
        sugg_gap = 18
        total_sugg_w = 3 * sugg_w + 2 * sugg_gap
        sugg_start_x = (self.screen_w - total_sugg_w) // 2
        sugg_y = 70

        for s_idx in range(3):
            bx = sugg_start_x + s_idx * (sugg_w + sugg_gap)
            btn = Button(
                pos=[bx, sugg_y],
                text="",
                size=[sugg_w, sugg_h],
                key_type="suggestion",
                key_id=f"SUGG_{s_idx}",
                radius=14
            )
            self.suggestion_buttons.append(btn)
            self.buttons.append(btn)

        # 2. Main Keyboard Rows
        # Row dimensions
        btn_w = 84
        btn_h = 56
        gap_x = 10
        gap_y = 10
        y_start = 130

        if not self.symbols_mode:
            # === QWERTY + Numbers Mode ===
            
            # Row 0: Numbers (1 2 3 4 5 6 7 8 9 0)
            row0_keys = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
            row0_w = len(row0_keys) * btn_w + (len(row0_keys) - 1) * gap_x
            row0_x = (self.screen_w - row0_w) // 2
            for j, key in enumerate(row0_keys):
                bx = row0_x + j * (btn_w + gap_x)
                by = y_start
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="number"))

            # Row 1: Q W E R T Y U I O P
            row1_keys = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"]
            if not self.caps_lock:
                row1_keys = [k.lower() for k in row1_keys]
            row1_w = len(row1_keys) * btn_w + (len(row1_keys) - 1) * gap_x
            row1_x = (self.screen_w - row1_w) // 2
            for j, key in enumerate(row1_keys):
                bx = row1_x + j * (btn_w + gap_x)
                by = y_start + (btn_h + gap_y)
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="letter"))

            # Row 2: A S D F G H J K L
            row2_keys = ["A", "S", "D", "F", "G", "H", "J", "K", "L"]
            if not self.caps_lock:
                row2_keys = [k.lower() for k in row2_keys]
            row2_w = len(row2_keys) * btn_w + (len(row2_keys) - 1) * gap_x
            row2_x = (self.screen_w - row2_w) // 2
            for j, key in enumerate(row2_keys):
                bx = row2_x + j * (btn_w + gap_x)
                by = y_start + 2 * (btn_h + gap_y)
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="letter"))

            # Row 3: CAPS, Z X C V B N M, BACK
            row3_letters = ["Z", "X", "C", "V", "B", "N", "M"]
            if not self.caps_lock:
                row3_letters = [k.lower() for k in row3_letters]
            caps_w = 110
            back_w = 110
            row3_w = caps_w + len(row3_letters) * btn_w + back_w + (len(row3_letters) + 1) * gap_x
            row3_x = (self.screen_w - row3_w) // 2
            by = y_start + 3 * (btn_h + gap_y)

            # CAPS button
            caps_btn = Button([row3_x, by], "CAPS", [caps_w, btn_h], key_type="toggle", key_id="CAPS")
            caps_btn.is_active = self.caps_lock
            self.buttons.append(caps_btn)

            # Letters Z to M
            curr_x = row3_x + caps_w + gap_x
            for key in row3_letters:
                self.buttons.append(Button([curr_x, by], key, [btn_w, btn_h], key_type="letter"))
                curr_x += btn_w + gap_x

            # BACK button
            self.buttons.append(Button([curr_x, by], "BACK", [back_w, btn_h], key_type="action", key_id="BACK"))

            # Row 4: ?123, COMMA, SPACE, DOT, CLEAR, ENTER
            sym_w = 95
            punct_w = 65
            space_w = 340
            clear_w = 95
            enter_w = 115
            row4_w = sym_w + punct_w + space_w + punct_w + clear_w + enter_w + 5 * gap_x
            row4_x = (self.screen_w - row4_w) // 2
            by = y_start + 4 * (btn_h + gap_y)

            curr_x = row4_x
            self.buttons.append(Button([curr_x, by], "?123", [sym_w, btn_h], key_type="toggle", key_id="MODE_SYM"))
            curr_x += sym_w + gap_x

            self.buttons.append(Button([curr_x, by], ",", [punct_w, btn_h], key_type="letter"))
            curr_x += punct_w + gap_x

            self.buttons.append(Button([curr_x, by], "SPACE", [space_w, btn_h], key_type="space", key_id="SPACE"))
            curr_x += space_w + gap_x

            self.buttons.append(Button([curr_x, by], ".", [punct_w, btn_h], key_type="letter"))
            curr_x += punct_w + gap_x

            self.buttons.append(Button([curr_x, by], "CLEAR", [clear_w, btn_h], key_type="action", key_id="CLEAR"))
            curr_x += clear_w + gap_x

            self.buttons.append(Button([curr_x, by], "ENTER", [enter_w, btn_h], key_type="action", key_id="ENTER"))

        else:
            # === Symbols Mode ===
            
            # Row 0: Symbols ! @ # $ % ^ & * ( )
            row0_keys = ["!", "@", "#", "$", "%", "^", "&", "*", "(", ")"]
            row0_w = len(row0_keys) * btn_w + (len(row0_keys) - 1) * gap_x
            row0_x = (self.screen_w - row0_w) // 2
            for j, key in enumerate(row0_keys):
                bx = row0_x + j * (btn_w + gap_x)
                by = y_start
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="letter"))

            # Row 1: - _ = + [ ] { } \ /
            row1_keys = ["-", "_", "=", "+", "[", "]", "{", "}", "\\", "/"]
            row1_w = len(row1_keys) * btn_w + (len(row1_keys) - 1) * gap_x
            row1_x = (self.screen_w - row1_w) // 2
            for j, key in enumerate(row1_keys):
                bx = row1_x + j * (btn_w + gap_x)
                by = y_start + (btn_h + gap_y)
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="letter"))

            # Row 2: ~ ` ; : ' " < > ?
            row2_keys = ["~", "`", ";", ":", "'", "\"", "<", ">", "?"]
            row2_w = len(row2_keys) * btn_w + (len(row2_keys) - 1) * gap_x
            row2_x = (self.screen_w - row2_w) // 2
            for j, key in enumerate(row2_keys):
                bx = row2_x + j * (btn_w + gap_x)
                by = y_start + 2 * (btn_h + gap_y)
                self.buttons.append(Button([bx, by], key, [btn_w, btn_h], key_type="letter"))

            # Row 3: ABC, 1 2 3 4 5 6 7, BACK
            row3_syms = ["1", "2", "3", "4", "5", "6", "7"]
            abc_w = 110
            back_w = 110
            row3_w = abc_w + len(row3_syms) * btn_w + back_w + (len(row3_syms) + 1) * gap_x
            row3_x = (self.screen_w - row3_w) // 2
            by = y_start + 3 * (btn_h + gap_y)

            # ABC button to switch back
            abc_btn = Button([row3_x, by], "ABC", [abc_w, btn_h], key_type="toggle", key_id="MODE_ABC")
            self.buttons.append(abc_btn)

            curr_x = row3_x + abc_w + gap_x
            for key in row3_syms:
                self.buttons.append(Button([curr_x, by], key, [btn_w, btn_h], key_type="number"))
                curr_x += btn_w + gap_x

            self.buttons.append(Button([curr_x, by], "BACK", [back_w, btn_h], key_type="action", key_id="BACK"))

            # Row 4: ABC, ,, SPACE, ., CLEAR, ENTER
            abc_toggle_w = 95
            punct_w = 65
            space_w = 340
            clear_w = 95
            enter_w = 115
            row4_w = abc_toggle_w + punct_w + space_w + punct_w + clear_w + enter_w + 5 * gap_x
            row4_x = (self.screen_w - row4_w) // 2
            by = y_start + 4 * (btn_h + gap_y)

            curr_x = row4_x
            self.buttons.append(Button([curr_x, by], "ABC", [abc_toggle_w, btn_h], key_type="toggle", key_id="MODE_ABC"))
            curr_x += abc_toggle_w + gap_x

            self.buttons.append(Button([curr_x, by], ",", [punct_w, btn_h], key_type="letter"))
            curr_x += punct_w + gap_x

            self.buttons.append(Button([curr_x, by], "SPACE", [space_w, btn_h], key_type="space", key_id="SPACE"))
            curr_x += space_w + gap_x

            self.buttons.append(Button([curr_x, by], ".", [punct_w, btn_h], key_type="letter"))
            curr_x += punct_w + gap_x

            self.buttons.append(Button([curr_x, by], "CLEAR", [clear_w, btn_h], key_type="action", key_id="CLEAR"))
            curr_x += clear_w + gap_x

            self.buttons.append(Button([curr_x, by], "ENTER", [enter_w, btn_h], key_type="action", key_id="ENTER"))

    def toggle_caps(self):
        """Toggles CAPS Lock state and refreshes letter cases."""
        self.caps_lock = not self.caps_lock
        self.rebuild_layout()

    def toggle_symbols(self):
        """Toggles between QWERTY and Symbols layouts."""
        self.symbols_mode = not self.symbols_mode
        self.rebuild_layout()

    def set_suggestions(self, suggestions: List[str]):
        """Updates text on the suggestion chips."""
        for idx, btn in enumerate(self.suggestion_buttons):
            if idx < len(suggestions):
                btn.text = suggestions[idx]
            else:
                btn.text = ""

    def get_button_at(self, x: int, y: int) -> Optional[Button]:
        """Finds the button containing point (x, y)."""
        for btn in self.buttons:
            if btn.contains(x, y):
                # Don't return empty suggestion chips
                if btn.key_type == "suggestion" and not btn.text:
                    continue
                return btn
        return None


def draw_keyboard_backdrop(img, y1: int = 58, y2: int = 460):
    """
    Renders an elegant glassmorphism floating backdrop card behind the keyboard
    to ensure perfect visual contrast regardless of room lighting.
    """
    h, w, _ = img.shape
    pad_x = 40
    rx1, rx2 = pad_x, w - pad_x
    ry1, ry2 = y1, y2

    # Draw semi-transparent dark container over the keyboard zone
    sub_img = img[ry1:ry2, rx1:rx2]
    card_overlay = sub_img.copy()
    
    # Fill with deep obsidian tint
    cv2.rectangle(card_overlay, (0, 0), (rx2 - rx1, ry2 - ry1), THEME["backdrop_bg"], -1)
    
    # Blend overlay with 82% opacity
    cv2.addWeighted(card_overlay, 0.82, sub_img, 0.18, 0, sub_img)
    
    # Draw subtle perimeter stroke
    draw_rounded_rect(img, (rx1, ry1), (rx2, ry2), THEME["backdrop_border"], radius=18, thickness=1)


def draw_text_output_card(img, text: str, y_top: int = 480, height: int = 210, 
                          hand_present: bool = True, caps_lock: bool = False, fps: int = 30):
    """
    Renders a premium modern glassmorphic output panel showing typed text,
    a blinking cursor, character count, and system status indicators.
    """
    h, w, _ = img.shape
    x1, x2 = 50, w - 50
    y1, y2 = y_top, y_top + height

    # Sub-region glassmorphism blend
    sub_img = img[y1:y2, x1:x2]
    card_overlay = sub_img.copy()
    cv2.rectangle(card_overlay, (0, 0), (x2 - x1, y2 - y1), THEME["backdrop_bg"], -1)
    cv2.addWeighted(card_overlay, 0.88, sub_img, 0.12, 0, sub_img)

    # Rounded outer stroke
    draw_rounded_rect(img, (x1, y1), (x2, y2), THEME["backdrop_border"], radius=14, thickness=1)

    # --- Header Bar inside Output Card ---
    # Status Dot & Hand Status
    status_color = THEME["status_emerald"] if hand_present else THEME["status_amber"]
    status_label = "HAND DETECTED" if hand_present else "SEARCHING HAND..."
    cv2.circle(img, (x1 + 25, y1 + 25), 5, status_color, -1, cv2.LINE_AA)
    cv2.putText(img, status_label, (x1 + 38, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, status_color, 1, cv2.LINE_AA)

    # Caps Lock Badge
    if caps_lock:
        cv2.putText(img, "[CAPS ON]", (x1 + 220, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, THEME["key_active_dot"], 1, cv2.LINE_AA)

    # Character count
    count_str = f"CHARS: {len(text)}"
    cv2.putText(img, count_str, (x2 - 130, y1 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.45, THEME["key_border_action"], 1, cv2.LINE_AA)

    # Divider line
    cv2.line(img, (x1 + 15, y1 + 42), (x2 - 15, y1 + 42), THEME["backdrop_border"], 1, cv2.LINE_AA)

    # --- Display Typed Text with Blinking Cursor ---
    # Blinking cursor state (500ms cycle)
    show_cursor = int(time.time() * 2) % 2 == 0
    display_text = text + ("|" if show_cursor else " ")

    # Support multiple lines or horizontal scroll if text exceeds width
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.85
    thickness = 2
    max_text_width = (x2 - x1) - 40

    # If text is too long for one line, show last fitting segment or wrap
    (tw, th), _ = cv2.getTextSize(display_text, font, font_scale, thickness)
    if tw > max_text_width:
        # Trim from beginning for display
        trimmed = display_text
        while len(trimmed) > 4:
            (curr_w, _), _ = cv2.getTextSize("..." + trimmed, font, font_scale, thickness)
            if curr_w <= max_text_width:
                display_text = "..." + trimmed
                break
            trimmed = trimmed[1:]

    # Text render position
    tx = x1 + 25
    ty = y1 + 105

    cv2.putText(
        img,
        display_text,
        (tx, ty),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA
    )

    # Bottom tip helper
    tip = "Gesture Tip: Pinch Thumb & Index finger over any key to type. Tap suggestions to complete."
    cv2.putText(img, tip, (x1 + 25, y2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, THEME["status_dim"], 1, cv2.LINE_AA)


def draw_cursor_reticle(img, pos: Tuple[int, int], pinch_progress: float = 0.0, is_clicked: bool = False):
    """
    Renders an elegant futuristic reticle at the user's index fingertip.
    Visualizes pinch readiness with a radial progress ring.
    """
    cx, cy = pos
    r_outer = 18
    r_inner = 4

    if is_clicked:
        reticle_color = THEME["key_bg_press"]
        cv2.circle(img, (cx, cy), r_outer + 4, reticle_color, 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), r_inner + 2, reticle_color, -1, cv2.LINE_AA)
    else:
        # Color based on pinch tightness
        reticle_color = lerp_color(THEME["key_text_normal"], THEME["key_border_hover"], pinch_progress)
        
        # Outer thin guide circle
        cv2.circle(img, (cx, cy), r_outer, (100, 110, 130), 1, cv2.LINE_AA)
        
        # Center target dot
        cv2.circle(img, (cx, cy), r_inner, reticle_color, -1, cv2.LINE_AA)
        
        # Dynamic pinch arc (fills from 0 to 360 degrees as pinch tightens)
        if pinch_progress > 0.05:
            angle = int(360 * min(1.0, pinch_progress))
            cv2.ellipse(img, (cx, cy), (r_outer, r_outer), -90, 0, angle, THEME["key_border_hover"], 2, cv2.LINE_AA)


def drawAll(img, buttonList: List[Button], dt: float = 0.033):
    """
    Draws all buttons in buttonList with modern styling.
    Preserves exact function name and arguments for full backward compatibility.
    """
    for button in buttonList:
        button.draw(img, dt=dt)
    return img
