import torch
from model import EncoderDecoder

def iou_score(pred, target, threshold=0.5, eps=1e-6):
    """IoU = Intersection over Union"""
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum() - intersection

    return (intersection + eps) / (union + eps)


def dice_score(pred, target, threshold=0.5, eps=1e-6):
    """Dice = 2 * |X ∩ Y| / (|X| + |Y|)"""
    pred_bin = (pred > threshold).float()

    intersection = (pred_bin * target).sum()
    total = pred_bin.sum() + target.sum()

    return (2 * intersection + eps) / (total + eps)

def load_model(checkpoint_path, device):
    model = EncoderDecoder().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model
