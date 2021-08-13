# same content as playground.ipynb
from PIL import Image
from dataset import rainy_dataset, show_tensor_image, show_multi_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import simple_energy
import torch
from model import simple_energy, ResNetModel
from train import single_forward, train

dataset=rainy_dataset(data_path="large_datasets/rainy_image_dataset",data_len=100, rainy_extent=3,normalize=True)
dataloader = DataLoader(dataset, batch_size = 5, shuffle=True)

ground_batch, rain_batch = next(iter(dataloader))
model = ResNetModel(args = None)
train(model, dataloader, steps=7, epochs=5, lr=1000, optimizer_lr=0.001,demo=True)