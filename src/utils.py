import numpy as np

def normalize_landmarks(hand_landmarks):
    """
    Normalizes hand landmarks: 
    1. Wrist-relative (Landmark 0 is the origin).
    2. Scaling (Max distance is 1.0).
    3. Flattening to a 1D list of 63 values.
    """
    # 1. Extract raw coordinates
    raw_landmarks = []
    for lm in hand_landmarks.landmark:
        raw_landmarks.append([lm.x, lm.y, lm.z])
    
    # Convert to numpy array for easier math
    landmarks_array = np.array(raw_landmarks)
    
    # 2. Wrist-Relative Shifting
    # Subtract the wrist (index 0) from all landmarks
    wrist = landmarks_array[0]
    normalized_array = landmarks_array - wrist
    
    # 3. Flatten to 1D (21 points * 3 dims = 63 values)
    flattened = normalized_array.flatten()
    
    # 4. Scaling
    # Find the maximum absolute value to scale everything between -1 and 1
    max_val = np.max(np.abs(flattened))
    if max_val != 0:
        flattened = flattened / max_val
        
    return flattened.tolist()