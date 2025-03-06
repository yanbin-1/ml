import torch
import torchvision.models as models

model = models.mobilenet_v2(pretrained=False)

x = torch.randn(1, 3, 224, 224)
model.forward(x)