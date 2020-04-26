import torch
import torchvision
from torchvision import datasets, models, transforms
from PIL import Image


from efficientnet_pytorch import EfficientNet




def efficientnet():
    model = EfficientNet.from_pretrained('efficientnet-b0')