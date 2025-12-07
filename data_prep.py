from torch.utils.data import Dataset
from PIL import Image
import torch
import os

import torchvision.transforms.v2 as transforms


class ImageStyleDataset(Dataset):
    def __init__(self, style_dir, image_dir, transform=None):
        super().__init__()
        self.style_dir = style_dir
        self.image_dir = image_dir
        self.style_paths = os.listdir(self.style_dir)
        self.image_paths = os.listdir(self.image_dir)
        self.transform = transform

    def __len__(self):
        return min(len(self.style_paths), len(self.image_paths))
    
    def __getitem__(self, index):
        style_path = os.path.join(self.style_dir, self.style_paths[index])
        image_path = os.path.join(self.image_dir, self.image_paths[index])
        style = Image.open(style_path)
        image = Image.open(image_path)
        if self.transform:
            style = self.transform(style)
            image = self.transform(image)
        return style, image
    

def init_cyclegan_transform(image_size, image_size_resized):
    transform = transforms.Compose([
        transforms.Resize(image_size_resized),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
        transforms.Normalize([0.5], [0.5])
    ])
    return transform