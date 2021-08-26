# demo the effect on one picture

import torch
from dataset import rainy_dataset, show_tensor_image, show_multi_image
from torch.utils.data import DataLoader
from train import single_forward
from ssim import ssim
from psnr import psnr
import numpy as np

# step is default 5
DEFAULT_STEP = 5
def model_demo(
    model_name="model_lr1_005",
    lr = 1,
    test_dataset="rainy_image_dataset",
    rainy_extent=3,
    image_store = "image_store"
    ):
    model = torch.load("model_saved/"+model_name+".pth")
    model = model.cpu()
    model.eval()
    dataset=rainy_dataset(data_path="large_datasets/"+test_dataset,data_subpath="testing",data_len=5, rainy_extent=rainy_extent,normalize=True)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)
    count =0
    for ground_batch, rainy_batch in dataloader:    
        fake_image_output = single_forward(rainy_batch,model,5,lr)[-1]
        
        save_folder = image_store
        save_name = model_name+"_"+str(count)
        show_tensor_image(rainy_batch[0],save=True,save_folder=save_folder,save_name=save_name+"_"+"rain.png")
        show_tensor_image(fake_image_output[0],save=True,save_folder=save_folder,save_name=save_name+"_"+"fake.png")
        show_tensor_image(ground_batch[0],save=True,save_folder=save_folder,save_name=save_name+"_"+"ground.png")

        count += 1
        

if __name__ =="__main__":
    model_demo()    






    