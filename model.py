import torch.nn as nn

class BasicCNN(nn.Module):
    def __init__(self, num_classes):
        super(BasicCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # (100x100x3) → (100x100x32)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → (50x50x32)

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # → (50x50x64)
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                          # → (25x25x64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),# → (25x25x128)
            nn.ReLU(),
            nn.MaxPool2d(2, 2)                           # → (12x12x128)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(12*12*128, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x