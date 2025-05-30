import torch
import sys
from PIL import Image
from torchvision import transforms
from model import BasicCNN
import os

def predict_image(image_filename, model_path="fruits_cnn.pth"):
    # Construct the full path to the image
    image_path = os.path.join("data", "examples", image_filename)
    
    # Check if the image file exists
    if not os.path.exists(image_path):
        print(f"❌ Error: Image file '{image_filename}' not found in data/examples/")
        return

    # Check if the model file exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found. Please train the model first.")
        return

    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    except Exception as e:
        print(f"❌ Error loading image: {str(e)}")
        return

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BasicCNN(num_classes=201)  # Update to match the saved model's number of classes
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    # Make prediction
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        
        # Get probabilities
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence = probabilities[0][predicted].item() * 100

    # Load class names from the dataset
    from dataset import get_dataloaders
    _, _, class_names = get_dataloaders("data/fruits-360_100x100/fruits-360", batch_size=1)
    
    predicted_class = class_names[predicted.item()]
    print(f"\n🔍 Prediction Results:")
    print(f"📸 Image: {image_filename}")
    print(f"🍎 Predicted Class: {predicted_class}")
    print(f"📊 Confidence: {confidence:.2f}%")

def main():
    if len(sys.argv) != 2:
        print("❌ Usage: python3 predict.py <image_filename>")
        print("Example: python3 predict.py apple.png")
        print("Note: Images should be placed in the data/examples/ directory")
        sys.exit(1)

    image_filename = sys.argv[1]
    predict_image(image_filename)

if __name__ == "__main__":
    main()