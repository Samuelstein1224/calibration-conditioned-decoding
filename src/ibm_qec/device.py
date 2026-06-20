"""Shared compute device for model training and evaluation."""
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
