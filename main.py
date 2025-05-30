import torch
import os
from dataset import get_dataloaders
from model import BasicCNN
from train import train_model
from evaluate import evaluate_model

def main():
    try:
        print("🚀 Starting the fruit classification model...")
        
        # Check if CUDA is available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"💻 Using device: {device}")
        
        print("📂 Loading dataset...")
        train_loader, test_loader, class_names = get_dataloaders("data/fruits-360_100x100/fruits-360", batch_size=32)
        print(f"✅ Dataset loaded successfully. Found {len(class_names)} classes")
        
        model = BasicCNN(num_classes=len(class_names))
        model_path = "fruits_cnn.pth"

        if os.path.exists(model_path):
            print("✅ Found saved model. Loading weights...")
            model.load_state_dict(torch.load(model_path))
        else:
            print("🚀 No saved model found. Training from scratch...")
            train_model(model, train_loader, device, epochs=5)
            torch.save(model.state_dict(), model_path)
            print(f"💾 Model saved to {model_path}")

        print("📊 Evaluating model...")
        evaluate_model(model, test_loader, class_names, device)
        
    except Exception as e:
        print(f"❌ An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()