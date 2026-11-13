"""Standalone inference API: no simulator or privileged navigation data required."""
import torch
from models.revision2 import PolicyV2,StudentV2

def load_policy(checkpoint,device='cuda'):
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if not state.get('inference_only'):raise ValueError('Use the inference-only model.pt export; keep last.pt for training resume')
    if state['config']['model'].get('architecture')!='revision2':raise ValueError('Unsupported model architecture')
    model=PolicyV2(state['config']['model'],state['mode']).to(device);model.load_state_dict(state['model'])
    return StudentV2(model,device)
