import cv2
import mediapipe as mp
import torch
import string
from src.model import ISLClassifier
from src.utils import normalize_landmarks # Import the exact same math used for training!

# 1. Configuration
MODEL_PATH = "models/best_isl_model.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LETTERS = list(string.ascii_uppercase)

def main():
    print("Loading Model...")
    model = ISLClassifier(input_size=126, num_classes=26).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    print("Model Loaded Successfully!")

    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.7)

    cap = cv2.VideoCapture(0)
    print("Starting Webcam... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the raw frame (do not flip yet)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            # Initialize 126 zeros (just like preprocess.py)
            feature_vector = [0.0] * 126
            
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw the skeleton
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # Get the handedness (Left or Right)
                hand_label = results.multi_handedness[i].classification[0].label
                
                # Normalize using YOUR exact function from utils.py
                normalized_data = normalize_landmarks(hand_landmarks)
                
                # Put the data in the correct slot!
                if hand_label == "Left":
                    feature_vector[0:63] = normalized_data
                else:
                    feature_vector[63:126] = normalized_data

            # Convert to Tensor
            input_tensor = torch.tensor([feature_vector], dtype=torch.float32).to(DEVICE)

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                _, predicted = torch.max(outputs.data, 1)
                predicted_letter = LETTERS[predicted.item()]

            # Flip the frame so it acts like a mirror for the user
            display_frame = cv2.flip(frame, 1)
            cv2.putText(display_frame, f"Sign: {predicted_letter}", (20, 70), 
                        cv2.FONT_HERSHEY_DUPLEX, 2, (0, 255, 0), 3)
            
            cv2.imshow('ISL Real-Time Translator', display_frame)

        else:
            display_frame = cv2.flip(frame, 1)
            cv2.imshow('ISL Real-Time Translator', display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()