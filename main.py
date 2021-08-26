# same content as playground.ipynb
from PIL import Image
from dataset import rainy_dataset, show_tensor_image, show_multi_image
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from model import simple_energy
import torch
from model import simple_energy, ResNetModel
from train import single_forward, train
from model_test import model_test


for lr in [1, 5, 10]:
    for optimizer_lr in [0.001,0.005]:
        dataset=rainy_dataset(data_path="large_datasets/rainy_image_dataset",data_len=900, rainy_extent=3,normalize=True)
        dataloader = DataLoader(dataset, batch_size = 10, shuffle=True)

        ground_batch, rain_batch = next(iter(dataloader))
        model = ResNetModel(args = None)

        optimizer_lr_str = ""
        if optimizer_lr == 0.001:
            optimizer_lr_str = "001"
        elif optimizer_lr == 0.005:
            optimizer_lr_str = "005"
        

        model_name = "CDmodel_lr"+str(lr)+"_"+optimizer_lr_str
        mse_losses, positive_energys, negative_energys,cd_losses, losses = train(
            model, 
            dataloader,
            steps=5, 
            epochs=30,
            from_model_name = "",
            lr=lr,
            optimizer_lr=optimizer_lr,
            demo=False,
            use_cuda=True,
            save_name = model_name
            )

        file = open("model_loss_record/"+model_name+"_loss_record_"+str(3),"w")
        for loss in losses:
            file.write(str(loss))
            file.write("\n")
        file.close()

        file = open("model_loss_record/"+model_name+"_MSEloss_record_"+str(3),"w")
        for loss in mse_losses:
            file.write(str(loss))
            file.write("\n")
        file.close()

        file = open("model_loss_record/"+model_name+"_CDloss_record_"+str(3),"w")
        for loss in cd_losses:
            file.write(str(loss))
            file.write("\n")
        file.close()

        file = open("model_loss_record/"+model_name+"_positive_energy_record_"+str(3),"w")
        for loss in positive_energys:
            file.write(str(loss))
            file.write("\n")
        file.close()

        file = open("model_loss_record/"+model_name+"_negative_energy_record_"+str(3),"w")
        for loss in negative_energys:
            file.write(str(loss))
            file.write("\n")
        file.close()

        model_test(
            model_name=model_name,
            lr=lr,
            # loss_criterion="ssim",
            test_dataset="rainy_image_dataset",
            rainy_extent=3,
            use_cuda = False
        )

        