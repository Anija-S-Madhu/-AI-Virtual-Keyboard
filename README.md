# AI Virtual Keyboard

An AI-powered virtual keyboard that allows users to type using hand gestures through a webcam.

## Overview

AI Virtual Keyboard is a computer vision-based application developed using Python, OpenCV, and MediaPipe. It detects hand movements and enables users to interact with a virtual keyboard without physically touching a keyboard.

## Features

- Real-time hand tracking using MediaPipe.
- Gesture-based virtual typing.
- Modern dark-themed keyboard interface.
- Smooth key selection and visual feedback.
- Word suggestions while typing.
- No automatic word correction.
- Backspace, Space, Enter, and Caps Lock support.
- Number and symbol keyboard.
- Improved gesture detection and typing accuracy.

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy

## Project Structure

AI_Virtual_Keyboard/
│
├── main.py
├── handTracker.py
├── keyboard.py
├── word_suggestions.py
├── test_keyboard_suite.py
├── requirements.txt
├── preview_ui.png
├── .gitignore
└── README.md


## Installation

1. Clone the repository.

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
python main.py
```

4. Allow webcam access when prompted.

## Testing

Run the test suite:

```bash
python test_keyboard_suite.py
```

## Future Improvements

- Enhanced gesture recognition.
- Improved word suggestion engine.
- Additional keyboard customization.
- Better typing accuracy and responsiveness.

## Developer

Developed as a Python and Computer Vision project.

---

**AI Virtual Keyboard – Touchless Typing Through Hand Gestures.**