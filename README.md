# VisionGuard AI

## Overview
VisionGuard AI is a real-time AI-powered vision system that processes live camera feeds, detects objects using YOLOv8, and provides AI-generated descriptions or alerts. The system can also respond to voice commands using Google Gemini AI.

## Features
- **Object Detection:** Utilizes YOLOv8 to detect and classify objects in real-time.
- **AI-Powered Scene Analysis:** Describes the detected scene using Google Gemini AI.
- **Voice Command Processing:** Listens for voice commands and responds accordingly.
- **Alert System:** Detects fast-moving objects and warns the user.
- **Text-to-Speech Output:** Communicates detected information through audio.

## Dependencies
- OpenCV (`cv2`)
- Ultralytics YOLO (`ultralytics`)
- Pyttsx3 (Text-to-Speech)
- SpeechRecognition (`speech_recognition`)
- NumPy (`numpy`)
- Google Generative AI (`google.generativeai`)

## Installation
1. Install the required Python packages:
   ```sh
   pip install opencv-python ultralytics pyttsx3 speechrecognition numpy google-generativeai
   ```
2. Download the YOLOv8 model:
   ```sh
   yolo predict --model yolov8s.pt --source 0
   ```

## Configuration
- Update the **RTSP camera URL** in `rtsp_url`.
- Set your **Google Gemini API key** in `GEMINI_API_KEY`.
- Adjust **priority objects** and alert sensitivity in `priority_objects` and `check_alerts`.

## Usage
1. Run the script:
   ```sh
   python visionguard_ai.py
   ```
2. The system will:
   - Detect objects in the live camera feed.
   - Describe the scene in real-time.
   - Respond to voice commands.
   - Provide warnings for fast-moving objects.

## Voice Commands
- "What do you see?"
- "Describe the scene."
- "Is there anything dangerous?"
- "Stop"

## Future Improvements
- Implement better NLP understanding for commands.
- Enhance speed detection for moving objects.
- Improve voice recognition accuracy.

---
### Author
Developed as part of an engineering project by a dedicated team focused on AI-powered vision systems.

