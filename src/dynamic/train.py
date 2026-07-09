# src/dynamic/train.py

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from src.dynamic.dataset import get_dynamic_loaders
from src.dynamic.model import ISLSequenceClassifier
from src.dynamic.standardization import TOTAL_FEATURES

# --- Configuration ---
PROCESSED_DIR = "data/processed_dynamic"
MODEL_SAVE_PATH = "models/best_isl_dynamic.pth"
BATCH_SIZE = 16
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 15
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_num_classes():
    labels_path = os.path.join(PROCESSED_DIR, "labels.json")
    with open(labels_path, "r") as f:
        labels = json.load(f)
    return len(labels)

def compute_class_weights(y_train_path):
    y_train = np.load(y_train_path)
    class_counts = np.bincount(y_train)
    total_samples = len(y_train)
    weights = total_samples / (len(class_counts) * class_counts)
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)

def train_model():
    print(f"Using Device: {DEVICE}")
    
    train_loader, val_loader, test_loader = get_dynamic_loaders(
        processed_dir=PROCESSED_DIR, 
        batch_size=BATCH_SIZE
    )
    
    num_classes = load_num_classes()
    print(f"Initializing baseline LSTM Model for {num_classes} classes...")
    
    model = ISLSequenceClassifier(
        input_size=TOTAL_FEATURES,   # Exactly 165
        hidden_size=128, 
        num_layers=1, 
        num_classes=num_classes, 
        dropout=0.3
    ).to(DEVICE)
    
    y_path = os.path.join(PROCESSED_DIR, "y_dynamic.npy")
    class_weights = compute_class_weights(y_path)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    print("\nStarting Training...")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss, correct_train, total_train = 0.0, 0, 0
        
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_train_loss += loss.item() * sequences.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_train_loss / total_train
        train_acc = (correct_train / total_train) * 100
        
        model.eval()
        running_val_loss, correct_val, total_val = 0.0, 0, 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * sequences.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = running_val_loss / total_val
        val_acc = (correct_val / total_val) * 100
        
        print(f"Epoch [{epoch+1:03d}/{EPOCHS}] Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= PATIENCE:
                print(f"\nEarly stopping triggered! No improvement for {PATIENCE} epochs.")
                break

    print("\nLoading Best Model for Final Testing...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    
    correct_test, total_test = 0, 0
    with torch.no_grad():
        for sequences, labels in test_loader:
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
            
    test_acc = (correct_test / total_test) * 100
    print(f"\n=== FINAL TEST ACCURACY: {test_acc:.2f}% ===")

if __name__ == "__main__":
    train_model()