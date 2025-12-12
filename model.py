import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SmallCNN(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, 1, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1, 1)
        self.fc1 = nn.Linear(16 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.adaptive_avg_pool2d(x, (8, 8))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

def get_model_weights(model):
    """(Helper) Flatten model weights into a 1D numpy array."""
    weights = []
    for p in model.parameters():
        weights.append(p.detach().cpu().numpy().ravel())
    return np.concatenate(weights)

def flatten_gradients(model):
    """(Helper) Flatten model gradients into a 1D numpy array."""
    arrs = []
    for p in model.parameters():
        if p.grad is not None:
            arrs.append(p.grad.view(-1).cpu().numpy())
        else:
            # Should not happen in normal training flow if loss.backward() was called
            arrs.append(np.zeros(p.numel(), dtype=np.float32))
    return np.concatenate(arrs)
