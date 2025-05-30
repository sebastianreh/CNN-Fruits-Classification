import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np

def train_improved_model(model, train_loader, test_loader, device, epochs=20):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)
    
    model.to(device)
    
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total
        
        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_accuracy = 100 * val_correct / val_total
        val_loss = val_loss / len(test_loader)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "fruits_cnn_improved.pth")
            print(f"✅ New best model saved with validation accuracy: {val_accuracy:.2f}%")
        
        train_losses.append(epoch_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
              f"Val Acc: {val_accuracy:.2f}%")
    
    print(f"\n🎯 Training completed! Best validation accuracy: {best_accuracy:.2f}%")
    return train_losses, val_accuracies 