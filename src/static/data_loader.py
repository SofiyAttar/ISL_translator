import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

class ISLDataset(Dataset):
    def __init__(self, X, y, augment=False):
        """
        Args:
            X: Features (Landmarks)
            y: Labels (Letters)
            augment: Boolean, whether to apply random noise
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        if self.augment:
            # Add a tiny bit of Gaussian Noise (0.005) 
            # This makes the model robust to shaky hands
            noise = torch.randn_like(features) * 0.005
            features = features + noise

        return features, label

def get_data_loaders(csv_path, batch_size=64):
    # 1. Load Data
    df = pd.read_csv(csv_path)
    
    # 2. Separate Features (X) and Labels (y)
    # Assuming the last column is 'label'
    X = df.iloc[:, :-1].values
    y_raw = df.iloc[:, -1].values

    # 3. Encode Labels (A-Z to 0-25)
    encoder = LabelEncoder()
    y = encoder.fit_transform(y_raw)
    num_classes = len(encoder.classes_)

    # 4. Triple Split (Stratified)
    # Split 1: 85% Train+Val, 15% Final Test Vault
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    # Split 2: From the 85%, take 15% for Validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.15, stratify=y_train_val, random_state=42
    )

    # 5. Create Dataset Objects
    train_dataset = ISLDataset(X_train, y_train, augment=True) # Augment training data
    val_dataset = ISLDataset(X_val, y_val, augment=False)
    test_dataset = ISLDataset(X_test, y_test, augment=False)

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print(f"Data Splitting Complete:")
    print(f"Total Samples: {len(df)}")
    print(f"Training: {len(train_dataset)} | Validation: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    return train_loader, val_loader, test_loader, num_classes