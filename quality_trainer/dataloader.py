from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import os
import numpy as np
import torch
    
def get_transform(resize_w, resize_h, is_train: bool = False):
    x_width = resize_w
    y_height = resize_h
    if is_train:
        border_size = 8
        transform = transforms.Compose([
            transforms.Resize((y_height+border_size, x_width+border_size)),
            v2.RandomHorizontalFlip(p=0.5), 
            v2.RandomResizedCrop(size=(y_height, x_width), antialias=True),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((y_height, x_width)),
            transforms.ToTensor(),
        ])
        
    return transform   


class ImageQualityDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        
        bn = os.path.basename(path)
        quality = int(bn[0:bn.find("_")])
        quality = torch.from_numpy(np.array([quality]))

        return image, quality


def get_dataloader(paths, resize_w, resize_h, batch_size, is_train):
    transform = get_transform(resize_w, resize_h, is_train)
    dataset = ImageQualityDataset(paths, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)
    
