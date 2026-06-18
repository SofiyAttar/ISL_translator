import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_data_loaders
from src.model import ISLClassifier
import os

# 1. Hyperparameters (L4: Keep these easily adjustable)
CSV_PATH = "data/processed/landmarks.csv"
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models/best_isl_model.pth"

def train_model():
    # 2. Load Data
    train_loader, val_loader, test_loader, num_classes = get_data_loaders(CSV_PATH, BATCH_SIZE)

    # 3. Initialize Model, Loss, and Optimizer
    # We assume 63 input features (21 landmarks * 3 coords)
    model = ISLClassifier(input_size=126, num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 4. Training Variables
    best_val_acc = 0.0
    patience = 7  # Early stopping: Stop if no improvement for 7 epochs
    counter = 0

    print(f"Starting Training on {DEVICE}...")

    for epoch in range(EPOCHS):
        # --- TRAINING PHASE ---
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        avg_train_loss = train_loss / len(train_loader)
        train_acc = 100 * correct_train / total_train

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        avg_val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct_val / total_val

        print(f"Epoch [{epoch+1}/{EPOCHS}] - Train Acc: {train_acc:.2f}% | Val Acc: {val_acc:.2f}% | Val Loss: {avg_val_loss:.4f}")

        # 5. Checkpointing & Early Stopping (The L4 Logic)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Best Model Saved! (Acc: {best_val_acc:.2f}%)")
            counter = 0 # Reset early stopping counter
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print("Training Complete. Evaluating on Test Set...")
    
    # 6. Final Evaluation on the "Vault" (Test Set)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total_test += labels.size(0)
            correct_test += (predicted == labels).sum().item()
    
    print(f"Final Test Accuracy (Unseen Data): {100 * correct_test / total_test:.2f}%")

if __name__ == "__main__":
    train_model()