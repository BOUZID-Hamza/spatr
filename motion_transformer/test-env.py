import torch
from vit_pytorch.vit_for_small_dataset import ViT
import numpy as np

model = ViT(
    nb_of_frames = 32,
    frame_size = 1024,
    num_classes = 6,
    dim = 1024,
    depth = 3,
    heads = [2,3,4],
    mlp_dim = 512,
    dropout = 0.1,
    emb_dropout = 0.1
)

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

num_params = count_parameters(model)
print(num_params, "M param")

img = torch.randn(1, 32, 1024)

preds = model(img) # (1, 1000)
