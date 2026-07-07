# src/dynamic/preprocess.py

import os
import cv2
import json
import numpy as np
import mediapipe as mp
from src.utils import normalize_landmarks

# 1. Configuration
SEQUENCE_LENGTH = 30  # Target sequence length (T)
NUM_FEATURES = 126    # 63 (Left hand) + 63 (Right hand)
VIDEO_DIR = os.path.join("data", "raw_videos")
SAVE_DIR = os.path.join("data", "processed_dynamic")

# Ensure the output directory exists
os.makedirs(SAVE_DIR, exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, 
    max_num_hands=2, 
    min_detection_confidence=0.7
)

def extract_landmarks_from_video(video_path):
    """
    Extracts normalized landmarks from a video file, frame-by-frame.
    Returns a sequence of shape (F, 126) where F is the original frame count.
    """
    cap = cv2.VideoCapture(video_path)
    video_sequence = []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        # Initialize a blank 126-dimension vector for this frame (0.0 if hands aren't detected)
        feature_vector = [0.0] * NUM_FEATURES
        
        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Retrieve left/right classification
                hand_label = results.multi_handedness[i].classification[0].label
                
                # Apply normalization math from utils.py
                normalized_data = normalize_landmarks(hand_landmarks)
                
                # Assign landmarks to appropriate hand slots
                if hand_label == "Left":
                    feature_vector[0:63] = normalized_data
                else:
                    feature_vector[63:126] = normalized_data
                    
        video_sequence.append(feature_vector)
        
    cap.release()
    return np.array(video_sequence)

def adjust_sequence_length(sequence, target_len=SEQUENCE_LENGTH):
    """
    Standardizes sequence length to exactly target_len (30 frames).
    - If longer: uniformly downsamples.
    - If shorter: pads with zero-vectors at the end.
    """
    f_count = len(sequence)
    if f_count == 0:
        return np.zeros((target_len, NUM_FEATURES))
        
    if f_count == target_len:
        return sequence
        
    if f_count > target_len:
        # Downsample uniformly across the video duration
        indices = np.linspace(0, f_count - 1, target_len).astype(int)
        return sequence[indices]
        
    if f_count < target_len:
        # Pad with zero feature vectors at the end of the sequence
        padding = np.zeros((target_len - f_count, NUM_FEATURES))
        return np.vstack((sequence, padding))

def main():
    # Identify class folders inside the video directory
    # Ignores hidden files (e.g. .DS_Store)
    class_folders = [d for d in os.listdir(VIDEO_DIR) 
                     if os.path.isdir(os.path.join(VIDEO_DIR, d)) and not d.startswith('.')]
    class_folders.sort()
    
    print(f"Found {len(class_folders)} classes: {class_folders}")
    
    # Save a clean label mapping (e.g., {"0": "1. loud", "1": "2. quiet"})
    label_map = {idx: name for idx, name in enumerate(class_folders)}
    with open(os.path.join(SAVE_DIR, "labels.json"), "w") as f:
        json.dump(label_map, f, indent=4)
    print("Saved labels.json to processed directory.")

    X = []
    y = []

    # Iterate through class directories and process video files
    for label_idx, class_name in enumerate(class_folders):
        class_path = os.path.join(VIDEO_DIR, class_name)
        
        # Gather all typical video formats
        video_files = [f for f in os.listdir(class_path) 
                       if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
        
        print(f"\nProcessing class '{class_name}' ({len(video_files)} videos)...")
        
        for v_file in video_files:
            video_path = os.path.join(class_path, v_file)
            
            # Step 1: Extract frame-by-frame landmarks
            raw_seq = extract_landmarks_from_video(video_path)
            
            # Step 2: Resize temporal size to exactly 30 frames
            fixed_seq = adjust_sequence_length(raw_seq, SEQUENCE_LENGTH)
            
            X.append(fixed_seq)
            y.append(label_idx)

    # Convert lists to structured NumPy arrays
    X_array = np.array(X, dtype=np.float32)  # Target Shape: [Total_Videos, 30, 126]
    y_array = np.array(y, dtype=np.int64)    # Target Shape: [Total_Videos]

    # Save arrays to processed folder
    np.save(os.path.join(SAVE_DIR, "X_dynamic.npy"), X_array)
    np.save(os.path.join(SAVE_DIR, "y_dynamic.npy"), y_array)
    
    print(f"\nPreprocessing Complete!")
    print(f"Saved: X_dynamic.npy with shape {X_array.shape}")
    print(f"Saved: y_dynamic.npy with shape {y_array.shape}")

if __name__ == "__main__":
    main()