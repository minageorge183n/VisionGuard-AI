import cv2
from ultralytics import YOLO
import pyttsx3
import time
import speech_recognition as sr
import numpy as np
from collections import Counter
import google.generativeai as genai

# Configure Gemini API for AI-based responses
GEMINI_API_KEY = "AIzaSyACzCuP8i8a4beKuX1m6rlMtkxEEc3gq_"
genai.configure(api_key=GEMINI_API_KEY)

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def gemini_llm(prompt):
    """
    Generate AI-based responses using Gemini.
    Parameters:
        prompt (dict): Contains command, objects, and scene description.
    Returns:
        str: AI-generated response.
    """
    
    command = prompt.get("command", "").strip().lower()
    objects_str = prompt.get("objects", "")
    scene = prompt.get("scene", "")

    system_message = "You’re an AI assistant that helps with vision-based tasks and general chatting."

    if command:  
        full_prompt = f"Command: {command}\n{objects_str}\n{scene}"
    else:
        full_prompt = f"{objects_str}\n{scene}" if objects_str or scene else prompt.get("text", "")

    # Generate AI response
    response = gemini_model.generate_content(
        full_prompt, generation_config={"max_output_tokens": 200}
    )
    
    return response.text.strip() if response else "I didn't understand that."

# RTSP camera feed URL
rtsp_url = "rtsp://admin:mac0168210002%23@192.168.1.9:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)

if not cap.isOpened():
    print("Error: Couldn’t connect to camera.")
    exit()

# Load YOLOv8 model for object detection
model = YOLO("/Users/mina/Developer/BMO 2.0/ultralytics/yolov8s.pt")

# Text-to-speech setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)

def speak(text):
    """Convert text to speech."""
    engine.say(text)
    engine.runAndWait()

# Voice command setup
recognizer = sr.Recognizer()
mic = sr.Microphone()

def listen_for_command():
    """
    Capture voice input and convert it into text.
    Returns:
        str: Recognized command or empty string on failure.
    """
    with mic as source:
        print("Listening for command...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        try:
            audio = recognizer.listen(source, timeout=3, phrase_time_limit=3)
            command = recognizer.recognize_google(audio).lower()
            print(f"Command: {command}")
            return command
        except (sr.UnknownValueError, sr.WaitTimeoutError, sr.RequestError):
            return ""

def describe_scene(objects):
    """
    Generate a textual description of the detected scene.
    Parameters:
        objects (list): List of detected object labels.
    Returns:
        str: Description of the scene.
    """
    if not objects:
        return "It’s clear ahead"
    count = Counter(objects)
    total = len(objects)
    if total > 5:
        return "a busy scene with lots of activity"
    elif total > 2:
        top_items = [f"{k} ({v})" for k, v in count.most_common(2)]
        return f"a few things around: {', '.join(top_items)}"
    else:
        return f"a {' and a '.join(count.keys())}"

def check_alerts(boxes, prev_boxes, frame_width):
    """
    Detect fast-moving objects and generate alerts.
    Parameters:
        boxes (list): Current frame detected objects.
        prev_boxes (list): Previous frame detected objects.
        frame_width (int): Width of the video frame.
    Returns:
        list: List of warning messages.
    """
    alerts = []
    if prev_boxes is not None and len(boxes) > 0 and len(prev_boxes) > 0:
        for box in boxes:
            label = model.names[int(box.cls)]
            if label in ["car", "bicycle", "person"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                curr_center = (x1 + x2) // 2
                for prev_box in prev_boxes:
                    px1, py1, px2, py2 = map(int, prev_box.xyxy[0].cpu().numpy())
                    prev_center = (px1 + px2) // 2
                    speed = abs(curr_center - prev_center)
                    if speed > frame_width * 0.1:
                        alerts.append(f"Warning: {label} moving quickly")
                        break
    return alerts

priority_objects = ["person", "dog", "car", "bicycle"]
last_spoken, last_objects, prev_boxes = time.time(), [], None
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Main processing loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to retrieve frame. Reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(rtsp_url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1000)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'H264'))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        continue

    # YOLOv8 inference for object detection
    results = model(frame)
    objects = [model.names[int(cls)] for cls in results[0].boxes.cls.cpu().numpy()]
    boxes = results[0].boxes

    # Scene description
    scene_desc = describe_scene(objects)
    objects_str = " and ".join(Counter(objects).keys()) if objects else ""

    # AI-generated response and speech output
    if set(objects) != set(last_objects) and time.time() - last_spoken > 5:
        prompt = {"objects": objects_str, "scene": scene_desc, "command": ""}
        response = gemini_llm(prompt)
        speak(response)
        last_spoken, last_objects = time.time(), objects.copy()

    # Alert detection
    alerts = check_alerts(boxes, prev_boxes, frame_width)
    if alerts and time.time() - last_spoken > 2:
        for alert in alerts:
            speak(alert)
        last_spoken = time.time()
    prev_boxes = boxes

    # Process voice command
    command = listen_for_command()
    if command:
        prompt = {"objects": objects_str, "scene": scene_desc, "command": command}
        response = gemini_llm(prompt)
        speak(response)
    elif "stop" in command:
        speak("Stopping.")
        break

    cv2.imshow("Feed", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
