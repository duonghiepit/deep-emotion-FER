import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from typing import Optional, Callable, Tuple

class PlainDataset(Dataset):
    def __init__(self, csv_file: str, img_dir: str, data_type: str, transform: Optional[Callable] = None):
        """
        Args:
            csv_file (str): Path to the CSV file with image paths and labels.
            img_dir (str): Directory with all the images.
            data_type (str): Type of data (e.g., 'train', 'test', 'val').
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.csv_file = pd.read_csv(csv_file)
        self.labels = self.csv_file['emotion'].values
        self.img_dir = img_dir
        self.data_type = data_type
        self.transform = transform

    def __len__(self) -> int:
        return len(self.csv_file)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(index):
            index = index.tolist()
        img_path = os.path.join(self.img_dir, f'{self.data_type}_{index}.jpg')
        #print(f'Đây là image path: {img_path}')
        try:
            img = Image.open(img_path)  # Convert to RGB to handle grayscale images
        except FileNotFoundError:
            raise FileNotFoundError(f"Image file not found: {img_path}")

        label = torch.tensor(self.labels[index], dtype=torch.long)

        if self.transform:
            img = self.transform(img)

        return img, label
    
def eval_data_dataloader(csv_file: str, img_dir: str, data_type: str, sample_number: int, transform: Optional[Callable] = None) -> None:
    if transform is None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.Normalize((0.5,), (0.5,)),
            transforms.ToTensor()
        ])
        
    dataset = PlainDataset(csv_file=csv_file, img_dir=img_dir, data_type=data_type, transform=transform)

    img, label = dataset[sample_number]
    print(f"Label: {label.item()}")
    
    img_np = img.permute(1, 2, 0).numpy()  # Convert CHW to HWC format
    plt.imshow(img_np)
    plt.title(f"Label: {label.item()}")
    plt.axis('off')
    plt.show()
