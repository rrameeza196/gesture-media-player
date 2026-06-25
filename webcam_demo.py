# webcam_demo.py  --  Run LOCALLY (not in Colab)
# pip install mediapipe opencv-python pyautogui scikit-learn
import cv2, numpy as np, pickle, time, os, urllib.request
import mediapipe as mp
from collections import deque, Counter
import pyautogui
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions
from mediapipe.tasks.python.core.base_options import BaseOptions

MODEL_PATH  = 'gesture_model.pkl'
CONFIDENCE_THRESHOLD = 0.40
SMOOTH_WIN  = 5
COOLDOWN    = 18   # frames between commands

HAND_MODEL = "hand_landmarker.task"
if not os.path.exists(HAND_MODEL):
    print("Downloading hand_landmarker model...")
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
        HAND_MODEL
    )

GESTURE_CLASSES = {
    0: {"name":"Open Palm",   "command":"Play/Pause", "key":"space"},
    1: {"name":"Fist",        "command":"Stop",       "key":"s"},
    2: {"name":"Thumbs Up",   "command":"Vol Up",     "key":"volumeup"},
    3: {"name":"Thumbs Down", "command":"Vol Down",   "key":"volumedown"},
    4: {"name":"Index Left",  "command":"Prev Track", "key":"left"},
    5: {"name":"Index Right", "command":"Next Track", "key":"right"},
    6: {"name":"Victory",     "command":"Mute",       "key":"volumemute"},
}

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5,
)
hand_landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17),
]

def extract_features(landmarks):
    coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
    coords -= coords[0]
    s = np.max(np.abs(coords)) + 1e-6
    return (coords / s).flatten()

def resolve_gesture(hlm, model_pred):
    thumb_tip = hlm[4]
    thumb_mcp = hlm[2]
    index_tip = hlm[8]
    index_mcp = hlm[5]
    thumb_extend = abs(thumb_tip.y - thumb_mcp.y) + abs(thumb_tip.x - thumb_mcp.x)
    index_extend = abs(index_tip.y - index_mcp.y) + abs(index_tip.x - index_mcp.x)
    is_thumb_dominant = thumb_extend > index_extend * 1.5
    if model_pred in (2, 3) or (model_pred in (4, 5) and is_thumb_dominant):
        if thumb_tip.y < thumb_mcp.y:
            return 2
        else:
            return 3
    if model_pred in (4, 5) or (model_pred in (2, 3) and not is_thumb_dominant):
        if index_tip.x < index_mcp.x:
            return 5
        else:
            return 4
    return model_pred

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: No webcam found (VideoCapture(0) failed).")
    exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

buf   = deque(maxlen=SMOOTH_WIN)
last  = None
cool  = 0

print("Gesture Media Player Active  |  Press Q to quit")
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    res   = hand_landmarker.detect(mp_img)
    h, w  = frame.shape[:2]

    label, cmd, color = "No Hand", "", (120,120,120)

    if res.hand_landmarks:
        for hlm in res.hand_landmarks:
            for i, j in HAND_CONNECTIONS:
                x1, y1 = int(hlm[i].x * w), int(hlm[i].y * h)
                x2, y2 = int(hlm[j].x * w), int(hlm[j].y * h)
                cv2.line(frame, (x1, y1), (x2, y2), (0, 220, 100), 2)
            for lm in hlm:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 5, (0, 180, 255), -1)

            thumb_mcp = hlm[2]
            by = int(thumb_mcp.y * h)
            cv2.line(frame, (0, by), (w, by), (255, 180, 0), 1, cv2.LINE_AA)
            cv2.putText(frame, "UP", (10, by - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 100), 1)
            cv2.putText(frame, "DOWN", (10, by + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 100, 220), 1)

            feat  = extract_features(hlm)
            pred  = model.predict([feat])[0]
            proba = model.predict_proba([feat])[0]
            conf  = float(np.max(proba))

            if conf < CONFIDENCE_THRESHOLD:
                label = "Uncertain"
                cmd = ""
                color = (0, 200, 255)
                last = None
            else:
                pred = resolve_gesture(hlm, pred)
                buf.append(pred)
                smooth = Counter(buf).most_common(1)[0][0]
                label  = GESTURE_CLASSES[smooth]["name"]
                cmd    = GESTURE_CLASSES[smooth]["command"]
                color  = (0,220,100)
                if cool == 0 and smooth != last:
                    try: pyautogui.press(GESTURE_CLASSES[smooth]["key"])
                    except: pass
                    last = smooth; cool = COOLDOWN
    else:
        last = None

    if cool > 0: cool -= 1

    cv2.rectangle(frame, (0,0), (w,45), (15,15,30), -1)
    cv2.putText(frame, "Touchless Media Player | ITU HCI CP3",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    cv2.rectangle(frame, (0,h-90),(w,h-50),(15,15,30),-1)
    cv2.putText(frame, f"Gesture: {label}",
                (10,h-62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    cv2.rectangle(frame, (0,h-50),(w,h),(15,15,30),-1)
    if cmd:
        cv2.putText(frame, f"  Command: {cmd}",
                    (10,h-18), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,220,100), 2)

    cv2.imshow("Gesture Media Player", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()