import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import os
from glob import glob
import matplotlib.pyplot as plt

class CamouflageDataset(Dataset):
    def __init__(self, image_paths, mask_paths, img_transform=None, mask_transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.img_transform = img_transform
        self.mask_transform = mask_transform

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx]).convert("L")  # single channel

        if self.img_transform:
            image = self.img_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)

        return image, mask  
    def __len__(self):
        return len(self.image_paths)
    
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# we first define the segmentation model to define the various different regions in the image and what class they belong to
class SegmentationModel(nn.Module):
    
    # num classes is equal to one because we will have two classes, background and foreground (the animal)
    def __init__(self, num_classes=1):
        super(SegmentationModel, self).__init__()
        # encoder layers
        # here we will use resnet50 to extract the features since this is much simpler and practical then training a feauture extractor ourselves
        # we use the weights from imagenet1k which contains 1000 classes and a bit more than one million images
        pre_trained_model = models.resnet50(weights="IMAGENET1K_V1")
        # for debugging
        print("Model loaded.")
        # we save each of the stages from resnet50 so we can use them later to encode and decode (with skip connection as well)
        self.stage1 = nn.Sequential(pre_trained_model.conv1, pre_trained_model.bn1, pre_trained_model.relu)
        self.stage2 = nn.Sequential(pre_trained_model.maxpool, pre_trained_model.layer1)    
        self.stage3 = pre_trained_model.layer2
        self.stage4 = pre_trained_model.layer3
        self.stage5 = pre_trained_model.layer4

        # decoder layers
        self.deconv1 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.dec_bn1 = nn.BatchNorm2d(1024)

        self.deconv2 = nn.ConvTranspose2d(1024 + 1024, 512, 2, stride=2)
        self.dec_bn2 = nn.BatchNorm2d(512)

        self.deconv3 = nn.ConvTranspose2d(512 + 512, 256, 2, stride=2)
        self.dec_bn3 = nn.BatchNorm2d(256)

        self.deconv4 = nn.ConvTranspose2d(256 + 256, 64, 2, stride=2)
        self.dec_bn4 = nn.BatchNorm2d(64)

        self.deconv5 = nn.ConvTranspose2d(64 + 64, 64, 2, stride=2)
        self.dec_bn5 = nn.BatchNorm2d(64)

        # output layer
        self.deconv_out = nn.Conv2d(64, num_classes, kernel_size=1)
        
        # activation layers
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):

        e1 = self.leaky_relu(self.stage1(x))  # 64, H/2
        e2 = self.leaky_relu(self.stage2(e1)) # 256, H/4
        e3 = self.leaky_relu(self.stage3(e2)) # 512, H/8
        e4 = self.leaky_relu(self.stage4(e3)) # 1024, H/16
        e5 = self.leaky_relu(self.stage5(e4)) # 2048, H/32

        #decoder with skip connections
        d1 = self.relu(self.dec_bn1(self.deconv1(e5)))  # 1024
        d1 = torch.cat([d1, e4], dim=1)                 # 1024 + 1024 = 2048

        d2 = self.relu(self.dec_bn2(self.deconv2(d1)))  # 512
        d2 = torch.cat([d2, e3], dim=1)                 # 512 + 512 = 1024

        d3 = self.relu(self.dec_bn3(self.deconv3(d2)))  # 256
        d3 = torch.cat([d3, e2], dim=1)                 # 256 + 256 = 512

        d4 = self.relu(self.dec_bn4(self.deconv4(d3)))  # 64
        d4 = torch.cat([d4, e1], dim=1)                 # 64 + 64 = 128

        d5 = self.relu(self.dec_bn5(self.deconv5(d4)))  # 64
        # here we don't pass the last layer through 
        out = self.deconv_out(d5)  # raw logits, shape [B, num_classes, H, W]

        return out


image_train_dir = "CAMO-V.1.0-CVIU2019/Images/Train"
image_test_dir  = "CAMO-V.1.0-CVIU2019/Images/Test"
mask_dir        = "CAMO-V.1.0-CVIU2019/GT"

train_images = sorted(glob(os.path.join(image_train_dir, "*.*")))
test_images  = sorted(glob(os.path.join(image_test_dir, "*.*")))
train_masks  = sorted(glob(os.path.join(mask_dir, "*.*")))
test_masks   = sorted(glob(os.path.join(mask_dir, "*.*")))

image_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor()
])

mask_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor() # Converts [H,W,1] to [1,H,W] and normalizes to [0, 1]
])

train_dataset = CamouflageDataset(train_images, train_masks, img_transform=image_transform, mask_transform=mask_transform)
test_dataset  = CamouflageDataset(test_images, test_masks, img_transform=image_transform, mask_transform=mask_transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SegmentationModel(num_classes=1).to(device)

# here we use this loss because we are dealing with raw logits instead of the layer passed through a nonlinear function
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)


# training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_idx, (images, masks) in enumerate(train_loader, start=1):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(train_loader)}], "
              f"Batch Loss: {loss.item():.4f}", end="\r")  # `end="\r"` overwrites the same line

    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}")


# inference

output_dir = "pred_overlays"
os.makedirs(output_dir, exist_ok=True)

model.eval()
with torch.no_grad():
    for idx, (imgs, masks) in enumerate(test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds = (torch.sigmoid(outputs) > 0.5).float()

        for i in range(imgs.size(0)):
            img = imgs[i].cpu()  # [3,H,W]
            pred_mask = preds[i,0].cpu()  # [H,W]

            # clone original image
            overlay = img.clone()

            # apply red overlay where prediction is 1
            overlay[0, pred_mask>0] = 1.0  # Red
            overlay[1, pred_mask>0] = 0.0  # Green
            overlay[2, pred_mask>0] = 0.0  # Blue

            # optionally concatenate original and overlay side by side
            combined = torch.cat([img, overlay], dim=2).permute(1,2,0)  # HWC

            # save to folder
            plt.imsave(os.path.join(output_dir, f"overlay_pred_{idx*test_loader.batch_size + i}.png"), combined)