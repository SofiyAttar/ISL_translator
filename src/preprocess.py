import os
import cv2
import mediapipe as mp
import pandas as pd
import json
from utils import normalize_landmarks

# 1. Setup MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=2, 
    min_detection_confidence=0.5
)

# 2. Paths
RAW_DATA_PATH = "data/raw"
PROCESSED_DATA_PATH = "data/processed"
CSV_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "landmarks.csv")
LABEL_MAP_OUTPUT = os.path.join(PROCESSED_DATA_PATH, "labels.json")

def process_dataset():
    data = []
    labels_map = {}
    
    # Create directory if it doesn't exist
    if not os.path.exists(PROCESSED_DATA_PATH):
        os.makedirs(PROCESSED_DATA_PATH)

    # Get sorted list of folders (classes)
    classes = sorted([f for f in os.listdir(RAW_DATA_PATH) if os.path.isdir(os.path.join(RAW_DATA_PATH, f))])
    
    # Create integer mapping for labels
    labels_map = {label: i for i, label in enumerate(classes)}
    print(f"Detected Classes: {labels_map}")

    for label_name, label_id in labels_map.items():
        folder_path = os.path.join(RAW_DATA_PATH, label_name)
        print(f"Processing class: {label_name}...")

        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                continue
            
            # MediaPipe needs RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image_rgb)

            if results.multi_hand_landmarks:
                # Initialize 126 zeros (63 for Hand 1, 63 for Hand 2)
                feature_vector = [0.0] * 126
                
                # Check handedness and landmarks
                for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    # Get label (Left or Right)
                    hand_label = results.multi_handedness[i].classification[0].label
                    
                    normalized_data = normalize_landmarks(hand_landmarks)
                    
                    # If Left, fill 0-62. If Right, fill 63-125
                    if hand_label == "Left":
                        feature_vector[0:63] = normalized_data
                    else:
                        feature_vector[63:126] = normalized_data
                
                # Append the 126 features + the integer label
                data.append(feature_vector + [label_id])

    # 3. Save to CSV
    columns = [f"point_{i}" for i in range(126)] + ["label"]
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(CSV_OUTPUT, index=False)
    
    # 4. Save labels mapping to JSON for later use in inference
    with open(LABEL_MAP_OUTPUT, "w") as f:
        json.dump(labels_map, f)

    print(f" Preprocessing complete. Saved {len(df)} samples to {CSV_OUTPUT}")

if __name__ == "__main__":
    process_dataset()