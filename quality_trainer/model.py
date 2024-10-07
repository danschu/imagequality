from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights

from torch.nn import Sequential, Linear, Sigmoid
from torch import load

def get_quality_model(device, weights_path=None):
    weights = None
    if weights_path is None:
        weights = MobileNet_V3_Large_Weights.DEFAULT
    model = mobilenet_v3_large(weights=weights)
    model.classifier = Sequential(
        model.classifier,
        Linear(1000, 32),
        Linear(32, 16),
        Linear(16, 1),
        Sigmoid()
    )
    
    if weights_path is not None:
        model.load_state_dict(load(weights_path, map_location=device, weights_only=True))
        
    model = model.to(device=device)
    model = model.eval()
    return model
