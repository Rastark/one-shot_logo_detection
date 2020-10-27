import torch
import torch.nn.functional as F
from tqdm import tqdm

def eval(model, loader, device):
    model.eval()
    
    return None