import cv2
import numpy as np
from torchvision import transforms

IMG_SIZE = 224


def load_image(path):
    img = cv2.imread(path)

    if img is None:
        return np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


train_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),

    transforms.ColorJitter(brightness=0.15, contrast=0.15),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),

    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])