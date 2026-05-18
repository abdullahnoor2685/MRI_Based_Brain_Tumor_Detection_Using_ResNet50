import torch
import numpy as np
from PIL import Image
from src.preprocessing import val_transform


def predict(model, image_path, device):

    model.eval()

    img = Image.open(image_path).convert("RGB")
    img = val_transform(np.array(img))

    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)

        conf, pred = torch.max(probs, 1)

    return pred.item(), conf.item()