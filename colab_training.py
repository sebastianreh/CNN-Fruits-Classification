# =============================================================================
# FRUIT CLASSIFICATION MODEL - GOOGLE COLAB TRAINING SCRIPT
# =============================================================================

# Install required packages
print("📦 Installing required packages...")
!pip install -q tqdm scikit-learn seaborn matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os
import zipfile
import requests
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import glob

# =============================================================================
# 1. SETUP AND DATA EXTRACTION
# =============================================================================

def setup_colab_environment():
    """Setup the Colab environment and check GPU availability"""
    print("🚀 Setting up Google Colab environment...")
    
    # Check GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print(f"✅ GPU available: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("⚠️  GPU not available, using CPU")
    
    return device

def extract_dataset():
    """Find and extract the dataset zip file"""
    print("📁 Looking for dataset zip file in current directory...")
    
    # Look for zip files in current directory
    zip_files = glob.glob("*.zip")
    
    if not zip_files:
        print("❌ No zip files found in current directory.")
        print("📋 Current directory contents:")
        !ls -la
        return False
    
    print(f"📦 Found zip files: {zip_files}")
    
    # Extract the first zip file (or you can modify this to select specific one)
    zip_file = zip_files[0]
    print(f"📂 Extracting {zip_file}...")
    
    try:
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # List contents first
            file_list = zip_ref.namelist()
            print(f"   Archive contains {len(file_list)} files")
            
            # Extract all files
            zip_ref.extractall('.')
            print("✅ Dataset extracted successfully!")
            
        # Show extracted contents
        print("📋 Extracted contents:")
        !ls -la
        
        return True
        
    except Exception as e:
        print(f"❌ Error extracting {zip_file}: {str(e)}")
        return False

# =============================================================================
# 2. IMPROVED MODEL ARCHITECTURE
# =============================================================================

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes):
        super(ImprovedCNN, self).__init__()
        self.features = nn.Sequential(
            # First block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.1),
            
            # Second block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2),
            
            # Third block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.3),
            
            # Fourth block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.4),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

# =============================================================================
# 3. DATA LOADING WITH AUGMENTATION
# =============================================================================

def get_improved_dataloaders(data_dir, batch_size=32):
    """Create dataloaders with improved augmentation"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((110, 110)),
        transforms.RandomCrop(100),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Test", transform=test_transform)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=2, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=2, pin_memory=True
    )

    return train_loader, test_loader, train_dataset.classes

# =============================================================================
# 4. FIND DATA DIRECTORY
# =============================================================================

def find_data_directory():
    """Find the correct data directory path"""
    possible_paths = [
        "data/fruits-360_100x100/fruits-360",
        "data/fruits-360",
        "fruits-360_100x100/fruits-360",
        "fruits-360",
        "CNN Fruits/data/fruits-360_100x100/fruits-360"
    ]
    
    print("🔍 Searching for data directory...")
    
    # First, let's see what we have
    print("📋 Current directory structure:")
    !find . -type d -name "*fruit*" -o -name "*360*" -o -name "Training" -o -name "Test" 2>/dev/null | head -20
    
    for path in possible_paths:
        if os.path.exists(path):
            training_path = os.path.join(path, "Training")
            test_path = os.path.join(path, "Test")
            
            if os.path.exists(training_path) and os.path.exists(test_path):
                print(f"✅ Found data directory: {path}")
                return path
    
    # If not found, let's search more broadly
    print("🔍 Searching for Training and Test directories...")
    training_dirs = []
    for root, dirs, files in os.walk('.'):
        if 'Training' in dirs and 'Test' in dirs:
            training_dirs.append(root)
    
    if training_dirs:
        data_path = training_dirs[0]
        print(f"✅ Found data directory: {data_path}")
        return data_path
    
    print("❌ Could not find data directory with Training and Test folders")
    return None

# =============================================================================
# 5. IMPROVED TRAINING FUNCTION
# =============================================================================

def train_improved_model(model, train_loader, test_loader, device, epochs=20):
    """Enhanced training with validation tracking"""
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 'min', patience=3, factor=0.5, verbose=True
    )
    
    model.to(device)
    
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    print(f"🏋️ Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        train_correct = 0
        train_total = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for inputs, labels in progress_bar:
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
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{100 * train_correct / train_total:.2f}%'
            })
        
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
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), "fruits_cnn_best.pth")
            print(f"✅ New best model saved! Validation accuracy: {val_accuracy:.2f}%")
        
        train_losses.append(epoch_loss)
        val_accuracies.append(val_accuracy)
        
        print(f"Epoch {epoch+1}: Train Loss: {epoch_loss:.4f}, "
              f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
    
    print(f"\n🎯 Training completed! Best validation accuracy: {best_accuracy:.2f}%")
    return train_losses, val_accuracies

# =============================================================================
# 6. EVALUATION FUNCTION
# =============================================================================

def evaluate_model(model, test_loader, class_names, device):
    """Evaluate the model and show results"""
    model.eval()
    all_preds = []
    all_labels = []

    print("📊 Evaluating model...")
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # Overall accuracy
    accuracy = 100 * sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    print(f"\n🎯 Overall Accuracy: {accuracy:.2f}%")
    
    # Show some class-wise accuracies
    unique_labels = set(all_labels)
    print(f"\n📈 Sample class accuracies (showing first 10 classes):")
    for i in sorted(unique_labels)[:10]:
        class_mask = [l == i for l in all_labels]
        class_preds = [p for p, m in zip(all_preds, class_mask) if m]
        class_labels = [l for l, m in zip(all_labels, class_mask) if m]
        if class_labels:
            class_acc = 100 * sum(p == l for p, l in zip(class_preds, class_labels)) / len(class_labels)
            print(f"   {class_names[i]}: {class_acc:.1f}%")

# =============================================================================
# 7. PLOTTING FUNCTIONS
# =============================================================================

def plot_training_history(train_losses, val_accuracies):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot training loss
    ax1.plot(train_losses)
    ax1.set_title('Training Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    # Plot validation accuracy
    ax2.plot(val_accuracies)
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# 8. MAIN TRAINING SCRIPT
# =============================================================================

def main():
    """Main training function"""
    try:
        # Setup environment
        device = setup_colab_environment()
        
        # Extract dataset if needed
        if not os.path.exists('data') and not any(os.path.exists(p) for p in ["fruits-360", "Training"]):
            if not extract_dataset():
                return
        
        # Find data directory
        data_path = find_data_directory()
        if not data_path:
            print("❌ Please make sure your zip file contains the correct directory structure")
            print("   Expected: data/fruits-360_100x100/fruits-360/ with Training/ and Test/ folders")
            return
        
        print("📂 Loading dataset...")
        train_loader, test_loader, class_names = get_improved_dataloaders(data_path, batch_size=64)
        print(f"✅ Dataset loaded! Found {len(class_names)} classes")
        print(f"📊 Training samples: {len(train_loader.dataset)}")
        print(f"📊 Test samples: {len(test_loader.dataset)}")
        
        # Create improved model
        model = ImprovedCNN(num_classes=len(class_names))
        print(f"🧠 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
        
        # Train the model
        train_losses, val_accuracies = train_improved_model(
            model, train_loader, test_loader, device, epochs=25
        )
        
        # Plot training history
        plot_training_history(train_losses, val_accuracies)
        
        # Load best model and evaluate
        model.load_state_dict(torch.load("fruits_cnn_best.pth"))
        evaluate_model(model, test_loader, class_names, device)
        
        # Download the trained model
        print("\n💾 Downloading trained model...")
        from google.colab import files
        files.download('fruits_cnn_best.pth')
        
        print("\n🎉 Training completed successfully!")
        print("📁 Your trained model has been downloaded as 'fruits_cnn_best.pth'")
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

# =============================================================================
# 9. RUN THE TRAINING
# =============================================================================

if __name__ == "__main__":
    main() 