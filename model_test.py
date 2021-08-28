import torch
from dataset import rainy_dataset, show_tensor_image, show_multi_image
from torch.utils.data import DataLoader
from train import single_forward
from ssim import ssim
from psnr import psnr
import numpy as np
# this function will generate a model_report

# step is default 5
DEFAULT_STEP = 5
def model_test(
    model_name,
    lr,
    # loss_criterion,
    test_dataset="rainy_image_dataset",
    rainy_extent=3,
    use_cuda = True
    ):
    model = torch.load("model_saved/"+model_name+".pth")
    model = model.cpu()
    model.eval()
    dataset=rainy_dataset(data_path="large_datasets/"+test_dataset,data_subpath="testing",data_len=100, rainy_extent=rainy_extent,normalize=True)
    dataloader = DataLoader(dataset, batch_size = 1, shuffle=True)
    losses_ssim = []
    losses_psnr = []
    for ground_batch, rainy_batch in dataloader:
        if use_cuda:
            model = model.cuda(0)
            ground_batch = ground_batch.cuda(0)
            rainy_batch = rainy_batch.cuda(0)

        fake_image_output = single_forward(rainy_batch,model,5,lr)[-1]
        
        
        losses_ssim.append(ssim(ground_batch,fake_image_output,window_size=5).detach().cpu().numpy())
        losses_psnr.append(psnr(ground_batch,fake_image_output).detach().cpu().numpy())

    loss_ssim_estimate = np.mean(losses_ssim)
    loss_psnr_estimate = np.mean(losses_psnr)

    file = open("model_report/evaluate_"+model_name+"_"+str(rainy_extent),"w")
    file.write("model name : "+model_name+"\n")
    file.write("rainy extent : "+str(rainy_extent)+"\n")
    file.write("criteria : "+"ssim"+"\n")
    file.write("loss evaluation : "+str(loss_ssim_estimate)+"\n")
    file.write("criteria : "+"psnr"+"\n")
    file.write("loss evaluation : "+str(loss_psnr_estimate)+"\n")
    file.close()
    print("finish evaluation")

if __name__ == "__main__":

    model_test(
        model_name="CDmodel_lr1_001",
        lr=1,
        test_dataset="rainy_image_dataset",
        rainy_extent=3,
        use_cuda = False
    )

    model_test(
        model_name="model_lr1_005",
        lr=1,
        test_dataset="rainy_image_dataset",
        rainy_extent=3,
        use_cuda = False
    )




    