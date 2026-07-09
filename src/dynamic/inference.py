# src/dynamic/inference.py

import os
import cv2
import json
import torch
import numpy as np
import mediapipe as mp
import torch.nn.functional as F

from src.dynamic.model import ISLSequenceClassifier
from src.dynamic.standardization import extract_and_standardize, TOTAL_FEATURES

# --- Configuration ---
MODEL_PATH = "models/best_isl_dynamic.pth"
LABELS_PATH = "data/processed_dynamic/labels.json"
TARGET_SEQ_LEN = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_labels():
    with open(LABELS_PATH, "r") as f:
        labels_dict = json.load(f)
    return {int(k): v for k, v in labels_dict.items()}

def adjust_sequence_length(sequence, target_len=TARGET_SEQ_LEN):
    f_count = len(sequence)
    if f_count == 0:
        return np.full((target_len, TOTAL_FEATURES), 0.0)
    if f_count == target_len:
        return np.array(sequence)
    if f_count > target_len:
        indices = np.linspace(0, f_count - 1, target_len).astype(int)
        return np.array(sequence)[indices]
    if f_count < target_len:
        last_frame = sequence[-1]
        padding = np.tile(last_frame, (target_len - f_count, 1))
        return np.vstack((sequence, padding))

def draw_holistic_landmarks(image, results, mp_drawing, mp_holistic):
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
        )
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def main():
    print("Loading labels and model...")
    labels_map = load_labels()
    num_classes = len(labels_map)

    # Initialize model with 165 inputs (TOTAL_FEATURES)
    model = ISLSequenceClassifier(
        input_size=TOTAL_FEATURES, 
        hidden_size=128, 
        num_layers=1, 
        num_classes=num_classes,
        dropout=0.3
    ).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    
    is_recording = False
    recorded_sequence = []
    current_word = "Waiting for input..."
    current_confidence = 0.0

    print("\nWebcam Active. Press 'r' to start/stop recording, 'q' to quit.")

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            rgb_frame.flags.writeable = True

            keypoints = extract_and_standardize(results)

            if is_recording:
                recorded_sequence.append(keypoints)

            display_frame = cv2.flip(frame, 1)
            draw_holistic_landmarks(display_frame, holistic.process(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)), mp_drawing, mp_holistic)

            if is_recording:
                cv2.circle(display_frame, (30, 40), 10, (0, 0, 255), -1)
                cv2.putText(display_frame, f"RECORDING ({len(recorded_sequence)} frames)", (50, 45), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(display_frame, "Press 'r' to Start Recording", (20, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                color = (0, 255, 0) if current_confidence > 0.6 else (0, 255, 255)
                text = f"Sign: {current_word} ({current_confidence*100:.0f}%)" if current_confidence > 0 else current_word
                cv2.putText(display_frame, text, (20, 450), cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)

            cv2.imshow('ISL Dynamic Word Translator', display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                if not is_recording:
                    is_recording = True
                    recorded_sequence = []
                    current_word = "..."
                    current_confidence = 0.0
                else:
                    is_recording = False
                    if len(recorded_sequence) > 0:
                        print(f"\nRecorded: {len(recorded_sequence)} frames over your sign.")
                        
                        fixed_sequence = adjust_sequence_length(recorded_sequence, TARGET_SEQ_LEN)
                        input_tensor = torch.tensor([fixed_sequence], dtype=torch.float32).to(DEVICE)

                        # --- ON-THE-FLY VELOCITY CALCULATION ---
                        delta = input_tensor[:, 1:] - input_tensor[:, :-1]
                        mask = (input_tensor[:, 1:] == 0.0) | (input_tensor[:, :-1] == 0.0)
                        delta[mask] = 0.0
                        velocity_tensor = torch.cat([torch.zeros(1, 1, TOTAL_FEATURES).to(DEVICE), delta], dim=1)

                        # --- FEED PURE VELOCITY TO MODEL ---
                        with torch.no_grad():
                            outputs = model(velocity_tensor)
                            probabilities = F.softmax(outputs, dim=1)
                            max_prob, predicted_idx = torch.max(probabilities, 1)
                            
                            current_confidence = max_prob.item()
                            current_word = labels_map[predicted_idx.item()].upper()

                            print(f"--- Velocity-Based Prediction Breakdown ---")
                            for idx, prob in enumerate(probabilities[0]):
                                print(f"{labels_map[idx]}: {prob.item()*100:.2f}%")
                            print("----------------------------")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()