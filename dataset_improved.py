import torch
from torchvision import datasets, transforms

def get_improved_dataloaders(data_dir, batch_size=32):
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.Resize((110, 110)),  # Slightly larger for random crop
        transforms.RandomCrop(100),     # Random crop to 100x100
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    # No augmentation for testing
    test_transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=train_transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Test", transform=test_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
                                             shuffle=True, num_workers=4, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
                                            shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, test_loader, train_dataset.classes 