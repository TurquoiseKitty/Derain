# same content as playground.ipynb
from PIL import Image
from dataset import rainy_dataset, show_tensor_image, show_multi_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torch
from ebm_model import ResNetModel
from model_train import single_forward, train
from model_test import test
import os

DEFAULT_RAINY_EXTENT = 3

def loss_plot(path,loss_file_name,losses):
    # get loss list
    plt.figure()
    plt.plot(losses)
    plt.savefig(path+"/"+loss_file_name+".png")


def complete_training(model_name,lr=1,optimizer_lr=0.001,lambda_cd=0.01, lambda_reg=1e-4,steps=5,from_model=""):
    dataset=rainy_dataset(data_path="large_datasets/rainy_image_dataset",data_len=900, rainy_extent=DEFAULT_RAINY_EXTENT,normalize=True)
    dataloader = DataLoader(dataset, batch_size = 10, shuffle=True)
    model = ResNetModel(args = None)

    mse_losses, positive_energys, negative_energys,cd_losses, losses = train(
        dataloader, "rainy_image_dataset", DEFAULT_RAINY_EXTENT, epochs=50, model_name=model_name, save_path="model_bundle", from_model_path=from_model, use_cuda=True, cuda_index=4, \
                            args={'steps':steps,'lr':lr,'optimizer_lr':optimizer_lr,'lambda_cd':lambda_cd,'lambda_reg':lambda_reg})

    # plot these losses
    directory = "model_bundle/"+model_name+"/loss_plot"
    if not os.path.exists(directory):
        os.makedirs(directory)
    loss_plot(directory,"mse_losses",mse_losses)
    loss_plot(directory,"positive_energys",positive_energys)
    loss_plot(directory,"negative_energys",negative_energys)
    loss_plot(directory,"cd_losses",cd_losses)
    loss_plot(directory,"losses",losses)

    testing_dataset=rainy_dataset(data_path="large_datasets/rainy_image_dataset",data_subpath="testing",data_len=50, rainy_extent=DEFAULT_RAINY_EXTENT,normalize=True)
    testing_dataloader = DataLoader(testing_dataset, batch_size = 1, shuffle=True)

    test("model_bundle/"+model_name,testing_dataloader,"rainy_image_dataset",DEFAULT_RAINY_EXTENT,lr,steps,use_cuda=True, cuda_index=5)

if __name__ == "__main__":
    for lambda_cd in [0.01,0.05,0.1]:
        for lr in [1,0.5,2]:
            optimizer_lr = 0.001
            lambda_cd_str = str(lambda_cd).split('.')[1]
            lambda_lr_str = str(lr) if lr >= 1 else str(lr).split('.')[1]
            optimizer_lr_str = str(optimizer_lr).split('.')[1]

            model_name="model_GroupNorm"+"_lamCD"+lambda_cd_str+"_lr"+lambda_lr_str+"_optlr"+optimizer_lr_str
            complete_training(model_name,lr=lr,optimizer_lr=optimizer_lr,lambda_cd=lambda_cd, lambda_reg=1e-4,steps=5,from_model="")


        