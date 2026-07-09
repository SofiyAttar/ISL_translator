# src/dynamic/standardization.py

import numpy as np

# Upgraded Pose List (13 landmarks * 3 = 39 features)
# Includes Eyes, Ears, and Mouth Corners for Blind, Deaf, and Quiet!
SELECTED_POSE_INDICES = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

# Total features: Pose (39) + Left Hand (63) + Right Hand (63) = 165
TOTAL_FEATURES = 165

def extract_and_standardize(results):
    # 1. POSE (REFERENCE SKELETON - SCALED BY SHOULDER WIDTH)
    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark
        
        # Anchor: Midpoint of Shoulders
        ls, rs = pose_landmarks[11], pose_landmarks[12]
        anchor_x, anchor_y, anchor_z = (ls.x + rs.x) / 2, (ls.y + rs.y) / 2, (ls.z + rs.z) / 2
        
        # Scale: Shoulder width
        pose_scale = np.sqrt((ls.x - rs.x)**2 + (ls.y - rs.y)**2 + (ls.z - rs.z)**2)
        pose_scale = pose_scale if pose_scale > 0.01 else 1.0 

        pose = []
        for idx in SELECTED_POSE_INDICES:
            lm = pose_landmarks[idx]
            pose.extend([
                (lm.x - anchor_x) / pose_scale,
                (lm.y - anchor_y) / pose_scale,
                (lm.z - anchor_z) / pose_scale
            ])
        pose = np.array(pose)
    else:
        pose = np.zeros(len(SELECTED_POSE_INDICES) * 3)

    # 2. LEFT HAND (SELF-SCALED BY PALM SIZE)
    if results.left_hand_landmarks:
        lh_landmarks = results.left_hand_landmarks.landmark
        wrist = lh_landmarks[0]
        mcp = lh_landmarks[9]
        lh_scale = np.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2 + (wrist.z - mcp.z)**2)
        lh_scale = lh_scale if lh_scale > 0.001 else 1.0
        
        lh = []
        for lm in lh_landmarks:
            lh.extend([
                (lm.x - wrist.x) / lh_scale,
                (lm.y - wrist.y) / lh_scale,
                (lm.z - wrist.z) / lh_scale
            ])
        lh = np.array(lh)
    else:
        lh = np.zeros(21 * 3)

    # 3. RIGHT HAND (SELF-SCALED BY PALM SIZE)
    if results.right_hand_landmarks:
        rh_landmarks = results.right_hand_landmarks.landmark
        wrist = rh_landmarks[0]
        mcp = rh_landmarks[9]
        rh_scale = np.sqrt((wrist.x - mcp.x)**2 + (wrist.y - mcp.y)**2 + (wrist.z - mcp.z)**2)
        rh_scale = rh_scale if rh_scale > 0.001 else 1.0
        
        rh = []
        for lm in rh_landmarks:
            rh.extend([
                (lm.x - wrist.x) / rh_scale,
                (lm.y - wrist.y) / rh_scale,
                (lm.z - wrist.z) / rh_scale
            ])
        rh = np.array(rh)
    else:
        rh = np.zeros(21 * 3)

    return np.concatenate([pose, lh, rh])