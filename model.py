import torch
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import torch.nn as nn

transform = transforms.Compose([
    transforms.Resize((224, 224)),   # resize images to 224x224
    transforms.ToTensor(),           # convert PIL image to tensor
    transforms.Normalize(            # normalize like ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

DATA_DIR = './data' 

train_dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)

print(f"Found classes: {train_dataset.classes}")
print(f"Number of images: {len(train_dataset)}")

# check that we are using the gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# loading training set
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

