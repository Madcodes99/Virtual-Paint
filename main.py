import cv2
import torch
import time
import os
import mediapipe as mp
import numpy as np
from model import Model

current_path = os.getcwd()

cam_number = 0
flip = True
min_conf = 0.75
max_hands = 1
model_path = r'C:\Users\madha\OneDrive\Desktop\SIH\AirSlate\120.pt'

pen_color = (255, 0, 0)
eraser_size = 50
pen_size = 5
intermediate_step_gap = 1

color_squares = [
    {"color": (255, 0, 0), "pos": (50, 50), "size": 50},   
    {"color": (0, 255, 0), "pos": (110, 50), "size": 50},  
    {"color": (0, 0, 255), "pos": (170, 50), "size": 50},  
    {"color": (0, 255, 255), "pos": (230, 50), "size": 50},
    {"color": (255, 0, 255), "pos": (290, 50), "size": 50} 
]

brush_squares = [
    {"size": 5, "display_size": 50, "pos": (1050, 50)},
    {"size": 10, "display_size": 50, "pos": (1110, 50)},
    {"size": 20, "display_size": 50, "pos": (1170, 50)}
]

cv2.namedWindow('control')
img = np.zeros((10, 600, 3), np.uint8)

button = [20, 60, 145, 460]

canvas = None

def process_click(event, x, y, flags, params):
    global canvas
    if event == cv2.EVENT_LBUTTONDOWN:
        if button[0] < y < button[1] and button[2] < x < button[3]:
            cv2.imwrite('Image_' + str(time.time()) + '.png', canvas)  
            img[:80, :] = (0, 0, 255)

cv2.setMouseCallback('control', process_click)
cap = cv2.VideoCapture(cam_number, cv2.CAP_DSHOW)

width = 1080
height = 720

cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=max_hands,
    min_detection_confidence=min_conf,
    min_tracking_confidence=min_conf
)
mp_draw = mp.solutions.drawing_utils

_lm_list = [
    mpHands.HandLandmark.WRIST,
    mpHands.HandLandmark.THUMB_CMC,
    mpHands.HandLandmark.THUMB_MCP,
    mpHands.HandLandmark.THUMB_IP,
    mpHands.HandLandmark.THUMB_TIP,
    mpHands.HandLandmark.INDEX_FINGER_MCP,
    mpHands.HandLandmark.INDEX_FINGER_DIP,
    mpHands.HandLandmark.INDEX_FINGER_PIP,
    mpHands.HandLandmark.INDEX_FINGER_TIP,
    mpHands.HandLandmark.MIDDLE_FINGER_MCP,
    mpHands.HandLandmark.MIDDLE_FINGER_DIP,
    mpHands.HandLandmark.MIDDLE_FINGER_PIP,
    mpHands.HandLandmark.MIDDLE_FINGER_TIP,
    mpHands.HandLandmark.RING_FINGER_MCP,
    mpHands.HandLandmark.RING_FINGER_DIP,
    mpHands.HandLandmark.RING_FINGER_PIP,
    mpHands.HandLandmark.RING_FINGER_TIP,
    mpHands.HandLandmark.PINKY_MCP,
    mpHands.HandLandmark.PINKY_DIP,
    mpHands.HandLandmark.PINKY_PIP,
    mpHands.HandLandmark.PINKY_TIP
]

def landmark_extract(hand_lms, mpHands):
    output_lms = []

    for lm in _lm_list:
        lms = hand_lms.landmark[lm]
        output_lms.append(lms.x)
        output_lms.append(lms.y)
        output_lms.append(lms.z)

    return output_lms

def is_position_out_of_bounds(position, top_left, bottom_right):
    return (
            top_left[0] < position[0] < bottom_right[0]
            and top_left[1] < position[1] < bottom_right[1]
    )

model = Model()
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval() 

action_map = {0: 'Draw', 1: 'Erase', 2: 'None'}

font = cv2.FONT_HERSHEY_SIMPLEX
fontScale = 1
fontColor = (255, 255, 255)
lineType = 4
circles = []

was_drawing_last_frame = False

canvas = np.zeros((height, width, 3), np.uint8)

while True:
    success, frame = cap.read()
    if not success:
        break

    if flip:
        frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    if canvas.shape[:2] != (h, w):
        canvas = np.zeros((h, w, 3), np.uint8)

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if not results.multi_hand_landmarks:
        was_drawing_last_frame = False
        cv2.putText(frame, 'No hand in frame', (w - 300, h - 50), font, fontScale, fontColor, lineType)
    else:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mpHands.HAND_CONNECTIONS)

            landmark_list = landmark_extract(hand_landmarks, mpHands)
            model_input = torch.tensor(landmark_list, dtype=torch.float).unsqueeze(0)
            action = action_map[torch.argmax(model.forward(model_input)).item()]
            cv2.putText(frame, f"Mode : {action}", (w - 300, h - 50), font, fontScale, fontColor, lineType)

            for square in color_squares:
                cv2.rectangle(canvas, square['pos'], (square['pos'][0] + square['size'], square['pos'][1] + square['size']), square['color'], -1)

            cv2.putText(canvas, 'Brush Size', (1050, 30), font, fontScale, fontColor, lineType)

            for square in brush_squares:
                center = (square['pos'][0] + square['display_size'] // 2, square['pos'][1] + square['display_size'] // 2)
                radius = square['display_size'] // 2 
                cv2.circle(canvas, center, radius, (255, 255, 255), 2) 

                label = ""
                if square['size'] == 5:
                    label = "S"  # Small
                elif square['size'] == 10:
                    label = "M"  # Medium
                elif square['size'] == 20:
                    label = "L"  # Large

                cv2.putText(canvas, label, (square['pos'][0] + 15, square['pos'][1] + 35), font, fontScale, fontColor, lineType)

            index_finger_pos = (
                int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].x * w),
                int(hand_landmarks.landmark[mpHands.HandLandmark.INDEX_FINGER_TIP].y * h)
            )

            for square in color_squares:
                top_left = square['pos']
                bottom_right = (top_left[0] + square['size'], top_left[1] + square['size'])
                if is_position_out_of_bounds(index_finger_pos, top_left, bottom_right):
                    pen_color = square['color']
                    break

            cv2.circle(frame, index_finger_pos, 10, (255, 255, 255), 2)

            for square in brush_squares:
                center = (square['pos'][0] + square['display_size'] // 2, square['pos'][1] + square['display_size'] // 2)
                radius = square['display_size'] // 2
                distance_to_center = ((index_finger_pos[0] - center[0]) ** 2 + (index_finger_pos[1] - center[1]) ** 2) ** 0.5
                if distance_to_center <= radius:
                    pen_size = square['size']
                    break

            if action == 'Draw':
                pos = index_finger_pos

                if was_drawing_last_frame:
                    prev_pos = circles[-1][0]
                    x_distance = pos[0] - prev_pos[0]
                    y_distance = pos[1] - prev_pos[1]
                    distance = (x_distance ** 2 + y_distance ** 2) ** 0.5
                    num_step_points = int(distance // intermediate_step_gap) - 1
                    if num_step_points > 0:
                        x_normalized = x_distance / distance
                        y_normalized = y_distance / distance
                        for i in range(1, num_step_points + 1):
                            step_pos_x = prev_pos[0] + int(x_normalized * i)
                            step_pos_y = prev_pos[1] + int(y_normalized * i)
                            step_pos = (step_pos_x, step_pos_y)
                            circles.append((step_pos, pen_color, pen_size))
                            cv2.circle(canvas, step_pos, pen_size, pen_color, -1)  
                        
                circles.append((pos, pen_color, pen_size))
                cv2.circle(canvas, pos, pen_size, pen_color, -1)  
                was_drawing_last_frame = True
            else:
                was_drawing_last_frame = False

            if action == 'Erase':
                eraser_mid = [
                    int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].x * w),
                    int(hand_landmarks.landmark[mpHands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
                ]
                bottom_right = (eraser_mid[0] + eraser_size, eraser_mid[1] + eraser_size)
                top_left = (eraser_mid[0] - eraser_size, eraser_mid[1] - eraser_size)
                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 5)

                circles = [
                    (position, pen_color, pen_size)
                    for position, pen_color, pen_size in circles
                    if not is_position_out_of_bounds(position, top_left, bottom_right)
                ]

                cv2.rectangle(canvas, top_left, bottom_right, (0, 0, 0), -1)

    frame_with_canvas = cv2.addWeighted(frame, 0.8, canvas, 0.2, 0)

    cv2.imshow('Canvas', canvas)  
    cv2.imshow('Frame', frame_with_canvas)  

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
