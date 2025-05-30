import torch
from torchvision import datasets, transforms

def get_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((100, 100)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])
    
    train_dataset = datasets.ImageFolder(root=f"{data_dir}/Training", transform=transform)
    test_dataset = datasets.ImageFolder(root=f"{data_dir}/Test", transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, train_dataset.classes