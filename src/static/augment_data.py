import pandas as pd
import numpy as np
import os

CSV_PATH = "data/processed/landmarks.csv"
AUGMENTED_CSV_PATH = "data/processed/landmarks_augmented.csv"

def mirror_data():
    print("Loading original dataset...")
    df = pd.read_csv(CSV_PATH)
    
    augmented_rows = []
    
    # Extract features (exclude label)
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    print(f"Original samples: {len(df)}")
    
    for i in range(len(X)):
        row = X[i]
        label = y[i]
        
        # Create an empty 126-length array for the mirrored hand
        mirrored_row = np.zeros(126)
        
        # Check if Left Hand exists (if sum of absolute values > 0)
        left_hand = row[0:63]
        if np.sum(np.abs(left_hand)) > 0:
            # Move Left to Right slot, and invert X coordinates
            for j in range(0, 63, 3):
                mirrored_row[63 + j] = left_hand[j] * -1.0     # X (Inverted)
                mirrored_row[63 + j + 1] = left_hand[j + 1]    # Y (Unchanged)
                mirrored_row[63 + j + 2] = left_hand[j + 2]    # Z (Unchanged)
                
        # Check if Right Hand exists
        right_hand = row[63:126]
        if np.sum(np.abs(right_hand)) > 0:
            # Move Right to Left slot, and invert X coordinates
            for j in range(0, 63, 3):
                mirrored_row[j] = right_hand[j] * -1.0         # X (Inverted)
                mirrored_row[j + 1] = right_hand[j + 1]        # Y (Unchanged)
                mirrored_row[j + 2] = right_hand[j + 2]        # Z (Unchanged)
                
        augmented_rows.append(mirrored_row.tolist() + [label])
        
    print("Mirroring complete. Combining datasets...")
    
    # Combine original and augmented data
    columns = [f"point_{i}" for i in range(126)] + ["label"]
    aug_df = pd.DataFrame(augmented_rows, columns=columns)
    
    final_df = pd.concat([df, aug_df], ignore_index=True)
    
    final_df.to_csv(AUGMENTED_CSV_PATH, index=False)
    print(f"Success! New dataset saved to {AUGMENTED_CSV_PATH}")
    print(f"Total samples now: {len(final_df)}")

if __name__ == "__main__":
    mirror_data()