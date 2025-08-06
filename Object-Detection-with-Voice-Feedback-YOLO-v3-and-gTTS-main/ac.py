import numpy as np
import time
import cv2
import os
import pytesseract

# Uncomment and set path if you're on Windows
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

yolo_path = "yolo"
confidence_threshold = 0.5
nms_threshold = 0.3

labelsPath = os.path.sep.join([yolo_path, "coco.names"])
LABELS = open(labelsPath).read().strip().split("\n")

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = os.path.sep.join([yolo_path, "yolov3.weights"])
configPath = os.path.sep.join([yolo_path, "yolov3.cfg"])

print("[INFO] Loading YOLO...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (H, W) = frame.shape[:2]
    ln = net.getLayerNames()
    layer_indexes = net.getUnconnectedOutLayers()
    ln = [ln[i - 1] for i in layer_indexes.flatten()]

    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    boxes, confidences, classIDs = [], [], []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > confidence_threshold:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)
    detected_objects = []

    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y, w, h) = boxes[i]
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            text = f"{LABELS[classIDs[i]]}: {confidences[i]:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            detected_object = LABELS[classIDs[i]]
            print(f"[DETECTED OBJECT] {detected_object}")
            detected_objects.append(detected_object)

    # TEXT DETECTION SECTION
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    extracted_text = pytesseract.image_to_string(gray)

    if extracted_text.strip():
        print(f"[TEXT DETECTED]\n{extracted_text.strip()}")
    elif detected_objects:
        print(f"[NO TEXT] Objects Detected: {', '.join(set(detected_objects))}")
    else:
        print("[NO TEXT OR OBJECTS DETECTED]")

    cv2.imshow("Real-Time Detection with Text Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
