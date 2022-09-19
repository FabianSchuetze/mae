from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import torch
import os
class ItoddDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.files = sorted(os.listdir(root_dir))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.files[idx])
        image = Image.open(img_name)

        if self.transform:
            sample = self.transform(image)

        return sample, sample #coud be any, class id in origianl, discarded
