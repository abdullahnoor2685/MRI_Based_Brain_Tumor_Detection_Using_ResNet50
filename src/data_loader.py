import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import cv2
import numpy as np

class BrainTumorDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "image_path"]
        label = self.df.loc[idx, "label_encoded"]

        img = cv2.imread(path)

        if img is None:
            img = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, (224, 224))

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


def get_dataloaders(train_df, val_df, test_df, train_transform, val_transform, batch_size=16):
    
    train_dataset = BrainTumorDataset(train_df, train_transform)
    val_dataset = BrainTumorDataset(val_df, val_transform)
    test_dataset = BrainTumorDataset(test_df, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader