# src/dynamic/dataset.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split

class ISLSequenceDataset(Dataset):
    def __init__(self, sequences, labels, is_training=False):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.is_training = is_training

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx].clone() # Shape: [30, 165]

        # --- PILLAR 2: DATA AUGMENTATION ---
        if self.is_training:
            # 1. DYNAMIC MIRRORING
            if torch.rand(1).item() > 0.5:
                seq[:, 0::3] = seq[:, 0::3] * -1.0
                mirrored_seq = seq.clone()
                
                # Pose Swaps: Eyes(3:9), Ears(9:15), Mouth(15:21), Shoulders(21:27), Elbows(27:33), Wrists(33:39)
                mirrored_seq[:, 3:6] = seq[:, 6:9]
                mirrored_seq[:, 6:9] = seq[:, 3:6]
                mirrored_seq[:, 9:12] = seq[:, 12:15]
                mirrored_seq[:, 12:15] = seq[:, 9:12]
                mirrored_seq[:, 15:18] = seq[:, 18:21]
                mirrored_seq[:, 18:21] = seq[:, 15:18]
                mirrored_seq[:, 21:24] = seq[:, 24:27]
                mirrored_seq[:, 24:27] = seq[:, 21:24]
                mirrored_seq[:, 27:30] = seq[:, 30:33]
                mirrored_seq[:, 30:33] = seq[:, 27:30]
                mirrored_seq[:, 33:36] = seq[:, 36:39]
                mirrored_seq[:, 36:39] = seq[:, 33:36]
                
                # Hand Swaps: Left Hand (39:102) <-> Right Hand (102:165)
                mirrored_seq[:, 39:102] = seq[:, 102:165]
                mirrored_seq[:, 102:165] = seq[:, 39:102]
                seq = mirrored_seq

            # 2. SPATIAL JITTER
            noise = torch.randn_like(seq) * 0.001 
            valid_mask = (seq != 0.0)   
            seq[valid_mask] += noise[valid_mask]
            
            # 3. TEMPORAL SHIFT
            if torch.rand(1).item() > 0.5:
                shift = torch.randint(-2, 3, (1,)).item()
                if shift > 0:
                    seq = torch.cat((seq[shift:], seq[-1:].repeat(shift, 1)))
                elif shift < 0:
                    seq = torch.cat((seq[:1].repeat(abs(shift), 1), seq[:shift]))

        # --- PILLAR 4: ON-THE-FLY VELOCITY CALCULATION ---
        delta = seq[1:] - seq[:-1] 
        mask = (seq[1:] == 0.0) | (seq[:-1] == 0.0)
        delta[mask] = 0.0
        velocity_seq = torch.cat([torch.zeros(1, 165), delta], dim=0) # Shape: [30, 165]

        # Returns PURE Velocity trajectory
        return velocity_seq, self.labels[idx]

def get_dynamic_loaders(processed_dir="data/processed_dynamic", batch_size=16, random_state=42):
    X = np.load(os.path.join(processed_dir, "X_dynamic.npy"))
    y = np.load(os.path.join(processed_dir, "y_dynamic.npy"))
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=random_state, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=random_state, stratify=y_temp
    )
    
    train_dataset = ISLSequenceDataset(X_train, y_train, is_training=True)
    val_dataset = ISLSequenceDataset(X_val, y_val, is_training=False)
    test_dataset = ISLSequenceDataset(X_test, y_test, is_training=False)
    
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[y_train]
    
    sampler = WeightedRandomSampler(
        weights=torch.DoubleTensor(sample_weights), 
        num_samples=len(sample_weights), 
        replacement=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader