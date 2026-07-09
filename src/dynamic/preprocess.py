# src/dynamic/preprocess.py

import os
import cv2
import json
import numpy as np
import mediapipe as mp

from src.dynamic.standardization import extract_and_standardize, TOTAL_FEATURES

# --- Configuration ---
SEQUENCE_LENGTH = 30
NUM_FEATURES = TOTAL_FEATURES  # This is dynamically pulled as 165
VIDEO_DIR = os.path.join("data", "raw_videos")
SAVE_DIR = os.path.join("data", "processed_dynamic")

os.makedirs(SAVE_DIR, exist_ok=True)
mp_holistic = mp.solutions.holistic

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(video_path)
    video_sequence = []
    
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: 
                break
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb_frame.flags.writeable = False
            results = holistic.process(rgb_frame)
            
            keypoints = extract_and_standardize(results)
            video_sequence.append(keypoints)
            
    cap.release()
    return np.array(video_sequence)

def adjust_sequence_length(sequence, target_len=SEQUENCE_LENGTH):
    f_count = len(sequence)
    if f_count == 0:
        return np.zeros((target_len, NUM_FEATURES))
    if f_count == target_len:
        return sequence
    if f_count > target_len:
        indices = np.linspace(0, f_count - 1, target_len).astype(int)
        return sequence[indices]
    if f_count < target_len:
        last_frame = sequence[-1]
        padding = np.tile(last_frame, (target_len - f_count, 1))
        return np.vstack((sequence, padding))

def main():
    class_folders = [d for d in os.listdir(VIDEO_DIR) 
                     if os.path.isdir(os.path.join(VIDEO_DIR, d)) and not d.startswith('.')]
    class_folders.sort()
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    label_map = {idx: name for idx, name in enumerate(class_folders)}
    with open(os.path.join(SAVE_DIR, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=4)

    X = []
    y = []

    for label_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(VIDEO_DIR, class_name)
        video_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"Processing '{class_name}' ({len(video_files)} videos)...")
        for v_file in video_files:
            video_path = os.path.join(class_path, v_file)
            raw_seq = extract_landmarks_from_video(video_path)
            fixed_seq = adjust_sequence_length(raw_seq, SEQUENCE_LENGTH)
            X.append(fixed_seq)
            y.append(label_idx)

    X_array = np.array(X, dtype=np.float32)
    y_array = np.array(y, dtype=np.int64)

    np.save(os.path.join(SAVE_DIR, "X_dynamic.npy"), X_array)
    np.save(os.path.join(SAVE_DIR, "y_dynamic.npy"), y_array)
    
    print(f"\nData rebuilt successfully! X shape: {X_array.shape}")

if __name__ == "__main__":
    main()