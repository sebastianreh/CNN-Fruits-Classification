import torch
import os
from dataset_improved import get_improved_dataloaders
from model_improved import ImprovedCNN
from train_improved import train_improved_model
from evaluate import evaluate_model

def main():
    try:
        print("🚀 Starting improved fruit classification model...")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Using device: {device}")
        
        print("📂 Loading dataset with augmentation...")
        train_loader, test_loader, class_names = get_improved_dataloaders(
            "data/fruits-360_100x100/fruits-360", batch_size=32
        )
        print(f"✅ Dataset loaded successfully. Found {len(class_names)} classes")
        
        model = ImprovedCNN(num_classes=len(class_names))
        model_path = "fruits_cnn_improved.pth"

        if os.path.exists(model_path):
            print("✅ Found saved improved model. Loading weights...")
            model.load_state_dict(torch.load(model_path))
        else:
            print("🚀 No saved model found. Training improved model from scratch...")
            train_improved_model(model, train_loader, test_loader, device, epochs=20)

        print("📊 Evaluating improved model...")
        evaluate_model(model, test_loader, class_names, device)
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main() 