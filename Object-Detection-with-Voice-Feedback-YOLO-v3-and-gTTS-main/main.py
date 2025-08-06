import numpy as np
import time
import cv2
import os
import pyttsx3
import threading
import queue
import sys

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# === YOLO Configuration ===
yolo_path = "yolo"
confidence_threshold = 0.5
nms_threshold = 0.3

# Load class labels
labelsPath = resource_path(os.path.join(yolo_path, "coco.names"))
LABELS = open(labelsPath).read().strip().split("\n")

# Assign random colors to labels
np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

# Load YOLO weights and config
weightsPath = resource_path(os.path.join(yolo_path, "yolov3.weights"))
configPath = resource_path(os.path.join(yolo_path, "yolov3.cfg"))

print("[INFO] Loading YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Speech-related settings
audio_queue = queue.Queue()
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Previous feedback state
prev_description = ""
last_audio_time = 0
audio_cooldown = 10  # seconds

# === Audio Thread ===
def speak_worker():
    while True:
        description = audio_queue.get()
        if description is None:
            break
        try:
            engine.say(description)
            engine.runAndWait()
        except Exception as e:
            print(f"[Audio Error] {e}")

threading.Thread(target=speak_worker, daemon=True).start()

# === Get YOLO output layer names ===
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# === Main Loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]
    zone_thresholds = [W // 3, (2 * W) // 3]  # Left, center, right zones

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(output_layers)

    boxes, confidences, classIDs = [], [], []
    object_zones = {"left": [], "center": [], "right": []}

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - width / 2)
                y = int(centerY - height / 2)

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            label = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            obj_label = LABELS[classIDs[i]]
            center_x = x + w // 2

            if center_x < zone_thresholds[0]:
                object_zones["left"].append(obj_label)
            elif center_x < zone_thresholds[1]:
                object_zones["center"].append(obj_label)
            else:
                object_zones["right"].append(obj_label)

    # === Generate Structured Voice Feedback Description ===
    descriptions = []
    zone_text_map = {
        "left": "on your left",
        "center": "in front of you",
        "right": "on your right"
    }

    for zone in ["left", "center", "right"]:
        if object_zones[zone]:
            unique = sorted(set(object_zones[zone]))
            count = len(unique)
            if count == 1:
                obj_text = unique[0]
            elif count == 2:
                obj_text = f"{unique[0]} and {unique[1]}"
            else:
                obj_text = ", ".join(unique[:-1]) + f", and {unique[-1]}"
            descriptions.append(f"{obj_text} {zone_text_map[zone]}")

    description = ". ".join(descriptions) + "." if descriptions else ""
    current_time = time.time()

    # === Speak only if cooldown has passed ===
    if description and (current_time - last_audio_time > audio_cooldown):
        print(f"[VOICE FEEDBACK] {description}")
        prev_description = description
        last_audio_time = current_time
        with audio_queue.mutex:
            audio_queue.queue.clear()
        audio_queue.put(description)

    # === Display Frame ===
    cv2.imshow("Real-Time Object Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# === Cleanup ===
cap.release()
cv2.destroyAllWindows()
audio_queue.put(None)  # Stop audio thread
