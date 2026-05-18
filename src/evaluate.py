import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report


def evaluate(model, loader, device):

    model.eval()

    preds_list = []
    labels_list = []

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            outputs = model(x)

            preds = torch.argmax(outputs, dim=1)

            preds_list.extend(preds.cpu().numpy())
            labels_list.extend(y.cpu().numpy())

    acc = accuracy_score(labels_list, preds_list)
    f1 = f1_score(labels_list, preds_list, average="macro")

    report = classification_report(labels_list, preds_list)

    return acc, f1, report